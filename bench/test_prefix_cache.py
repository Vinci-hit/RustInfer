"""Test prefix caching effectiveness.

Sends requests with a SHARED long system prompt + different short user queries.
Compares with/without prefix caching to measure speedup.

Usage:
    # With prefix caching (server started with --enable-prefix-caching)
    python3 bench/test_prefix_cache.py --label with-cache

    # Without prefix caching (server started without the flag)
    python3 bench/test_prefix_cache.py --label no-cache
"""
import asyncio
import aiohttp
import json
import time
import argparse
from typing import List


# Long shared system prompt (~200 tokens) that all requests share
SYSTEM_PROMPT = """You are an expert AI assistant specializing in software engineering, computer science, and mathematics. You have deep knowledge of algorithms, data structures, system design, distributed systems, machine learning, and programming languages including Python, Rust, C++, Java, and Go. When answering questions, provide detailed explanations with code examples when appropriate. Always consider edge cases, performance implications, and best practices. If a question is ambiguous, ask for clarification. Format your responses with clear headings and bullet points for readability."""

# Different user queries (short, varied)
USER_QUERIES = [
    "What is a binary search tree?",
    "Explain TCP vs UDP in one paragraph.",
    "Write a Python function to reverse a linked list.",
    "What is the time complexity of quicksort?",
    "Explain what a mutex is.",
    "How does garbage collection work in Go?",
    "What is the CAP theorem?",
    "Write a Rust function to find duplicates in a vector.",
    "Explain the difference between stack and heap memory.",
    "What is consistent hashing?",
    "How do B-trees work?",
    "Explain async/await in Python.",
    "What is a bloom filter?",
    "Write a SQL query to find the second highest salary.",
    "What is the difference between process and thread?",
    "Explain how TLS handshake works.",
    "What is eventual consistency?",
    "Write a function to detect a cycle in a linked list.",
    "Explain the reactor pattern.",
    "What is copy-on-write?",
]


async def send_request(session, url, system, query, max_tokens):
    payload = {
        "model": "llama3.2-1b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(
        f"{url}/v1/chat/completions", json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        body = await resp.json()
    elapsed = time.perf_counter() - start
    if resp.status != 200:
        return {"success": False, "error": str(body), "latency": elapsed}
    usage = body["usage"]
    text = body["choices"][0]["message"]["content"]
    return {
        "success": True,
        "query": query,
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "latency": elapsed,
        "text": text[:100],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3, help="Number of rounds (prefix reuse happens in round 2+)")
    ap.add_argument("--label", default="test")
    args = ap.parse_args()

    print(f"=== Prefix Cache Test: {args.label} ===")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars (~200 tokens)")
    print(f"Queries: {len(USER_QUERIES)} unique, {args.rounds} rounds")
    print(f"Concurrency: {args.concurrency}")
    print()

    async with aiohttp.ClientSession() as session:
        # Warmup
        await send_request(session, args.url, "Hi", "Hello", 5)

        all_results = []
        for round_num in range(args.rounds):
            round_start = time.perf_counter()
            tasks = []
            for q in USER_QUERIES:
                tasks.append(send_request(session, args.url, SYSTEM_PROMPT, q, args.max_tokens))
                # Limit concurrency
                if len(tasks) >= args.concurrency:
                    batch_results = await asyncio.gather(*tasks)
                    all_results.extend(batch_results)
                    tasks = []
            if tasks:
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
            round_elapsed = time.perf_counter() - round_start

            round_results = all_results[round_num * len(USER_QUERIES):(round_num + 1) * len(USER_QUERIES)]
            successful = [r for r in round_results if r["success"]]
            if successful:
                avg_lat = sum(r["latency"] for r in successful) / len(successful)
                total_tok = sum(r["completion_tokens"] for r in successful)
                print(f"  Round {round_num + 1}: {len(successful)}/{len(USER_QUERIES)} ok, "
                      f"avg_lat={avg_lat:.3f}s, total_tok={total_tok}, "
                      f"wall={round_elapsed:.2f}s, throughput={total_tok/round_elapsed:.0f} tok/s")
            else:
                print(f"  Round {round_num + 1}: ALL FAILED")
                if round_results:
                    print(f"    {round_results[0].get('error', '')[:100]}")

    # Summary
    successful = [r for r in all_results if r["success"]]
    if not successful:
        print("\nALL FAILED")
        return

    latencies = sorted(r["latency"] for r in successful)
    total_completion = sum(r["completion_tokens"] for r in successful)

    print(f"\n{'═' * 50}")
    print(f"  Summary: {args.label}")
    print(f"{'═' * 50}")
    print(f"  Requests:    {len(successful)} ok / {len(all_results) - len(successful)} failed")
    print(f"  Total tok:   {total_completion}")
    print(f"  Latency p50: {latencies[len(latencies)//2]:.3f}s")
    print(f"  Latency p90: {latencies[int(len(latencies)*0.9)]:.3f}s")
    print(f"  Latency mean:{sum(latencies)/len(latencies):.3f}s")
    print(f"  Avg output:  {total_completion/len(successful):.0f} tok/req")
    print()
    print(f"  Round 1 = cold (no cache hit)")
    print(f"  Round 2+ = warm (prefix cache should hit for system prompt)")
    print(f"  If round 2/3 faster than round 1 → prefix cache working!")

    # Save
    out = f"/tmp/prefix_cache_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "results": all_results[:30]}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
