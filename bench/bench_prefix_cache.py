"""Smoke benchmark for RustInfer prefix caching.

This sends sequential requests with a long shared prompt prefix and different
suffixes. With `enable_prefix_caching = true`, the first request should populate
the RadixTree cache and later requests should reuse the shared prefix.

Usage:
    python3 bench/bench_prefix_cache.py --url http://127.0.0.1:8000
    python3 bench/bench_prefix_cache.py --endpoint chat --rounds 5
    python3 bench/bench_prefix_cache.py --dump-json /tmp/prefix_cache.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8000"


@dataclass
class RequestResult:
    index: int
    suffix: str
    status: int | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    text_preview: str
    error: str

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RustInfer prefix-cache reuse over HTTP")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument(
        "--endpoint",
        choices=["completions", "chat"],
        default="completions",
        help="API endpoint to exercise. completions gives exact raw prompt control.",
    )
    parser.add_argument("--model", default="qwen3", help="Model field sent in the request")
    parser.add_argument("--rounds", type=int, default=6, help="Total sequential requests")
    parser.add_argument("--prefix-lines", type=int, default=80, help="Shared prefix length")
    parser.add_argument("--max-tokens", type=int, default=1, help="Keep decode short for prefix tests")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dump-json", default="", help="Write raw results to this JSON file")
    return parser.parse_args()


def build_shared_prefix(lines: int) -> str:
    rows = [
        "You are reading a deterministic benchmark document.",
        "Every request in this script shares this prefix exactly.",
        "The suffix at the end changes to avoid a full-prompt cache hit.",
        "Summarize only the final task after the shared document.",
        "",
    ]
    for i in range(lines):
        rows.append(
            f"Shared prefix row {i:03d}: RustInfer prefix caching should reuse "
            "the KV entries produced for this repeated context."
        )
    rows.append("")
    return "\n".join(rows)


def build_prompt(shared_prefix: str, index: int) -> str:
    return (
        shared_prefix
        + f"Unique suffix {index:03d}: answer with the single word prefix-test-{index:03d}."
    )


def make_payload(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    common: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    if args.endpoint == "chat":
        return (
            "/v1/chat/completions",
            {
                **common,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    return (
        "/v1/completions",
        {
            **common,
            "prompt": prompt,
        },
    )


def parse_response(endpoint: str, body: dict[str, Any]) -> tuple[int, int, int, str]:
    usage = body.get("usage") or {}
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    if endpoint == "chat":
        text = ((choice.get("message") or {}).get("content") or "").strip()
    else:
        text = (choice.get("text") or "").strip()
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("total_tokens") or 0),
        text.replace("\n", "\\n")[:160],
    )


def post_json(url: str, path: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, json.loads(raw)


def send_one(args: argparse.Namespace, shared_prefix: str, index: int) -> RequestResult:
    suffix = f"prefix-test-{index:03d}"
    prompt = build_prompt(shared_prefix, index)
    path, payload = make_payload(args, prompt)
    start = time.perf_counter()
    try:
        status, body = post_json(args.url, path, payload, args.timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        prompt_tokens, completion_tokens, total_tokens, preview = parse_response(args.endpoint, body)
        error = "" if status == 200 else json.dumps(body, ensure_ascii=False)[:300]
        return RequestResult(
            index=index,
            suffix=suffix,
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            text_preview=preview,
            error=error,
        )
    except HTTPError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        body = e.read().decode("utf-8", errors="replace")
        return RequestResult(index, suffix, e.code, latency_ms, 0, 0, 0, "", body[:300])
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(index, suffix, None, latency_ms, 0, 0, 0, "", str(e)[:300])


def summarize(results: list[RequestResult]) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print()
    print("=" * 72)
    print("Prefix cache sequential test")
    print("=" * 72)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(
            f"#{r.index:02d} {status:4} "
            f"latency={r.latency_ms:9.1f}ms "
            f"prompt={r.prompt_tokens:5d} "
            f"completion={r.completion_tokens:3d} "
            f"text={r.text_preview!r}"
        )
        if r.error:
            print(f"     error={r.error}")

    if not ok:
        print("\nAll requests failed.")
        return

    first = ok[0]
    reuse = ok[1:]
    if reuse:
        reuse_latencies = [r.latency_ms for r in reuse]
        reuse_mean = statistics.mean(reuse_latencies)
        reuse_min = min(reuse_latencies)
        speedup = first.latency_ms / reuse_mean if reuse_mean > 0 else 0.0
        print()
        print(f"Cold request latency: {first.latency_ms:.1f}ms")
        print(f"Reuse mean latency:   {reuse_mean:.1f}ms")
        print(f"Reuse min latency:    {reuse_min:.1f}ms")
        print(f"Cold/reuse ratio:     {speedup:.2f}x")
    print(f"Successful requests: {len(ok)}/{len(results)}")
    if failed:
        print(f"Failed requests:     {len(failed)}")


def main() -> None:
    args = parse_args()
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2")
    if args.prefix_lines < 1:
        raise SystemExit("--prefix-lines must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")

    shared_prefix = build_shared_prefix(args.prefix_lines)
    print(f"Target:   {args.url}")
    print(f"Endpoint: /v1/{'chat/completions' if args.endpoint == 'chat' else 'completions'}")
    print(f"Rounds:   {args.rounds} sequential requests")
    print(f"Prefix:   {args.prefix_lines} shared text rows, {len(shared_prefix)} chars")
    print(f"Decode:   max_tokens={args.max_tokens}, temperature={args.temperature}")

    results = []
    for i in range(args.rounds):
        result = send_one(args, shared_prefix, i)
        results.append(result)
        print(f"sent #{i:02d}: status={result.status} latency={result.latency_ms:.1f}ms")
        if not result.ok:
            break

    summarize(results)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": vars(args),
                    "results": [asdict(r) for r in results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nWrote JSON: {args.dump_json}")


if __name__ == "__main__":
    main()
