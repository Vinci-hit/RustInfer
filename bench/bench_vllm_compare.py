"""Benchmark vLLM with identical settings for fair comparison against RustInfer.

Same model (Llama-3.2-1B), same prompts, same max_tokens=512, same batch sizes.
Saves results to /tmp/bench_vllm_results.json for side-by-side comparison.

Usage:
    python3 bench/bench_vllm_compare.py
"""
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List

from vllm import LLM, SamplingParams


@dataclass
class Result:
    prompt: str
    completion_tokens: int
    latency_s: float
    text: str


def run_batch(llm, prompts, max_tokens):
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    # Build prompts with SAME template as RustInfer (fixed date: 26 Jul 2024)
    # NOTE: no <|begin_of_text|> here — vLLM's tokenizer adds BOS automatically
    formatted = []
    for p in prompts:
        text = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n"
            "<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted.append(text)

    t0 = time.perf_counter()
    outputs = llm.generate(formatted, params)
    wall = time.perf_counter() - t0
    results = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        results.append(Result(
            prompt=prompts[i][:200],
            completion_tokens=n_tok,
            latency_s=wall,
            text=text,
        ))
    return wall, results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b")
    ap.add_argument("--prompts", default="/root/RustInfer/bench/bench_prompts.json")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--label", default="vllm-llama3.2-1b")
    ap.add_argument("--output", default="/tmp/bench_vllm_results.json")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA Graph (for comparison)")
    args = ap.parse_args()

    with open(args.prompts) as f:
        pool = json.load(f)
    long_prompts = [p for p in pool if isinstance(p, str) and len(p) >= 80]
    print(f"loaded {len(pool)} prompts ({len(long_prompts)} long)")

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        enforce_eager=args.enforce_eager,
    )
    print(f"vLLM ready (enforce_eager={args.enforce_eager})")

    # Warmup
    params = SamplingParams(max_tokens=16, temperature=0)
    warmup_text = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    llm.generate([warmup_text], params)
    print("warmup done")

    batch_sizes = [int(b) for b in args.batches.split(",")]
    all_results: List[dict] = []
    rows: List[dict] = []

    for bs in batch_sizes:
        prompts = long_prompts[:bs]
        wall, results = run_batch(llm, prompts, args.max_tokens)
        total_completion = sum(r.completion_tokens for r in results)
        agg_thru = total_completion / wall
        per_req_tok_s = total_completion / bs / wall  # all finish together in vLLM

        row = dict(
            label=args.label,
            batch=bs,
            max_tokens=args.max_tokens,
            wall_s=wall,
            total_completion=total_completion,
            agg_throughput_tok_s=agg_thru,
            per_req_mean_tok_s=per_req_tok_s,
        )
        rows.append(row)
        print(
            f"  batch={bs:>2}  wall={wall:6.2f}s  total_tok={total_completion:>5d}"
            f"  agg={agg_thru:7.1f} tok/s  per-req={per_req_tok_s:6.1f}"
        )
        for r in results:
            all_results.append(asdict(r))

    print()
    print(f"=== Summary: {args.label} (max_tokens={args.max_tokens}) ===")
    print(f"{'batch':>6} {'agg tok/s':>11} {'per-req tok/s':>14}")
    for r in rows:
        print(f"{r['batch']:>6} {r['agg_throughput_tok_s']:>11.1f} {r['per_req_mean_tok_s']:>14.1f}")

    output_data = {
        "summary": rows,
        "samples": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {args.output}")

    # Quick quality check
    print(f"\n=== Sample outputs (first 3) ===")
    for i, r in enumerate(all_results[:3]):
        print(f"\n--- Sample {i+1} ({r['completion_tokens']} tokens) ---")
        print(f"Prompt: {r['prompt'][:100]}")
        print(f"Output: {r['text'][:300]}...")


if __name__ == "__main__":
    main()
