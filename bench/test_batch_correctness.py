"""Test: single vs batch decode correctness.

Sends the same prompt as single request, then as batch=2 concurrent.
Compares outputs character by character to find divergence point.

Usage:
    python3 bench/test_batch_correctness.py
"""
import asyncio
import aiohttp


URL = "http://localhost:8000"
PROMPT = "Write a short story in third person narration about a protagonist who has to make an important career decision."
MAX_TOKENS = 100


async def send(session, prompt, max_tokens):
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with session.post(
        f"{URL}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        body = await resp.json()
    return body["choices"][0]["message"]["content"]


async def main():
    print(f"Prompt: {PROMPT[:80]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    # 1. Single request (baseline)
    async with aiohttp.ClientSession() as s:
        single = await send(s, PROMPT, MAX_TOKENS)
    print(f"Single ({len(single)} chars): {single[:120]}")

    # 2. Batch=2 (same prompt, concurrent)
    async with aiohttp.ClientSession() as s:
        r0, r1 = await asyncio.gather(
            send(s, PROMPT, MAX_TOKENS),
            send(s, PROMPT, MAX_TOKENS),
        )
    print(f"Batch0 ({len(r0)} chars): {r0[:120]}")
    print(f"Batch1 ({len(r1)} chars): {r1[:120]}")

    # 3. Batch=4
    async with aiohttp.ClientSession() as s:
        results = await asyncio.gather(*[send(s, PROMPT, MAX_TOKENS) for _ in range(4)])
    print(f"Batch4[0] ({len(results[0])} chars): {results[0][:120]}")
    print(f"Batch4[3] ({len(results[3])} chars): {results[3][:120]}")

    # Compare
    print("\n=== Comparison ===")
    all_outputs = [single, r0, r1] + results
    labels = ["Single", "Batch2[0]", "Batch2[1]", "Batch4[0]", "Batch4[1]", "Batch4[2]", "Batch4[3]"]

    ref = single
    for i, (label, output) in enumerate(zip(labels[1:], all_outputs[1:]), 1):
        if output == ref:
            print(f"  {label}: MATCH")
        else:
            # Find divergence point
            diverge = -1
            for j in range(min(len(ref), len(output))):
                if ref[j] != output[j]:
                    diverge = j
                    break
            if diverge == -1:
                diverge = min(len(ref), len(output))
            print(f"  {label}: DIVERGE at char {diverge}")
            print(f"    ref:    ...{ref[max(0,diverge-5):diverge+30]}")
            print(f"    actual: ...{output[max(0,diverge-5):diverge+30]}")


if __name__ == "__main__":
    asyncio.run(main())
