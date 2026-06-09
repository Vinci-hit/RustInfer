"""
Warm-up script for RustInfer Instruct interface.

Sends warm-up requests in concurrent batches, waits 3 seconds,
then sends one test batch and checks every returned result.

Usage:
    python bench/warmup_instruct.py --url http://localhost:8000 --warmup 1 --batch-size 32
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from typing import Any

import aiohttp


PROMPT = "写一段冒泡排序"


def parse_args():
    parser = argparse.ArgumentParser(description="Warm-up script for RustInfer Instruct interface")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dump-json", type=str, default="")
    parser.add_argument("--print-all", action="store_true")
    return parser.parse_args()


def extract_text(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    return choice.get("text") or choice.get("message", {}).get("content", "") or ""


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def inspect_results(label: str, results: list[dict[str, Any]], print_all: bool = False) -> None:
    total = len(results)
    ok_results = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]

    texts = [extract_text(r["body"]) for r in ok_results]
    hashes = [short_hash(t) for t in texts]
    hash_counts = Counter(hashes)

    token_counts = [
        (r["body"].get("usage", {}) or {}).get("completion_tokens")
        for r in ok_results
        if isinstance(r["body"], dict)
    ]
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

    print(f"{label}: {len(ok_results)}/{total} succeeded, unique_outputs={len(hash_counts)}")

    if latencies:
        print(
            f"  latency_ms: min={min(latencies):.1f} max={max(latencies):.1f} "
            f"avg={sum(latencies) / len(latencies):.1f}"
        )

    if token_counts:
        print(
            f"  completion_tokens: min={min(token_counts)} max={max(token_counts)} "
            f"unique={sorted(set(token_counts))}"
        )

    if failed:
        print("  Failed responses:")
        for r in failed:
            print(f"    #{r['index']:02d} status={r['status']} error/body={r['body']}")

    if ok_results:
        majority_hash, majority_count = hash_counts.most_common(1)[0]
        print(f"  majority_hash={majority_hash} count={majority_count}")
        for r, text, h in zip(ok_results, texts, hashes):
            usage = r["body"].get("usage", {}) if isinstance(r["body"], dict) else {}
            is_mismatch = h != majority_hash
            if print_all or is_mismatch:
                tag = "MISMATCH" if is_mismatch else "OK"
                preview = text.replace("\n", "\\n")[:240]
                print(
                    f"    [{tag}] #{r['index']:02d} "
                    f"latency={r['latency_ms']:.1f}ms "
                    f"hash={h} "
                    f"completion_tokens={usage.get('completion_tokens', 'N/A')} "
                    f"text={preview}"
                )


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    index: int,
) -> dict[str, Any]:
    payload = {
        "model": "qwen3",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            try:
                body = await resp.json()
            except Exception:
                body = await resp.text()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "index": index,
                "status": resp.status,
                "body": body,
                "latency_ms": elapsed_ms,
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "index": index,
            "status": None,
            "body": str(e),
            "latency_ms": elapsed_ms,
        }


async def run_batch(
    session: aiohttp.ClientSession,
    url: str,
    batch_size: int,
    max_tokens: int,
    label: str,
    print_all: bool,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    tasks = [send_request(session, url, PROMPT, max_tokens, i) for i in range(batch_size)]
    results = await asyncio.gather(*tasks)
    batch_elapsed_ms = (time.perf_counter() - start) * 1000

    inspect_results(label, results, print_all=print_all)
    print(f"  batch_wall_latency_ms: {batch_elapsed_ms:.1f}")

    return results


async def main():
    args = parse_args()
    all_results: list[dict[str, Any]] = []

    print(f"Target: {args.url}/v1/completions")
    print(f"Warm-up: {args.warmup} rounds x {args.batch_size} concurrent requests")
    print(f"Test: 1 round x {args.batch_size} concurrent requests")
    print(f"Prompt: {PROMPT}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    async with aiohttp.ClientSession(trust_env=False) as session:
        print("=" * 60)
        print("Warm-up phase")
        print("=" * 60)

        for i in range(args.warmup):
            results = await run_batch(
                session=session,
                url=args.url,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                label=f"Warm-up round {i + 1}",
                print_all=args.print_all,
            )
            all_results.append({"phase": "warmup", "round": i + 1, "results": results})

        print()
        print("Waiting 3 seconds...")
        await asyncio.sleep(3)

        print()
        print("=" * 60)
        print("Test batch")
        print("=" * 60)

        test_results = await run_batch(
            session=session,
            url=args.url,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            label="Test batch",
            print_all=True,
        )
        all_results.append({"phase": "test", "round": 1, "results": test_results})

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Dumped all responses to {args.dump_json}")

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
