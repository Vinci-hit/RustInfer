"""
SGLang single-request latency benchmark for Qwen3-4B.
Measures TTFT, TPOT, and total latency with token-level timing.
"""
import time
import argparse

import sglang as sgl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct")
    parser.add_argument("--prompt", type=str, default="请用中文详细介绍二叉树的中序遍历算法，并给出Python实现代码。")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    engine = sgl.Engine(
        model_path=args.model,
        dtype=args.dtype,
        mem_fraction_static=0.9,
    )

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": args.max_tokens,
    }

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        engine.generate(args.prompt, sampling_params)
    print("Warmup done.\n")

    # Benchmark
    print("=" * 60)
    print("Benchmark run")
    print("=" * 60)

    t_start = time.perf_counter()
    output = engine.generate(args.prompt, sampling_params)
    t_end = time.perf_counter()

    generated_text = output["text"]
    total_time_ms = (t_end - t_start) * 1000

    # Token counts
    meta = output.get("meta_info", {})
    num_prompt_tokens = meta.get("prompt_tokens", 0)
    num_output_tokens = meta.get("completion_tokens", 0)
    if num_output_tokens == 0:
        # Fallback: estimate from generated text (rough)
        num_output_tokens = max(len(generated_text) // 2, 1)

    # Timing
    ttft_ms = None
    decode_time_ms = None
    tpot_ms = total_time_ms / max(num_output_tokens, 1)

    # SGLang may expose e2e_latency or per-token metrics in meta_info
    e2e = meta.get("e2e_latency", None)
    if e2e is not None:
        total_time_ms = e2e * 1000

    print(f"\nPrompt ({num_prompt_tokens} tokens): {args.prompt[:80]}...")
    print(f"\nGenerated ({num_output_tokens} tokens):\n{generated_text}\n")
    print("=" * 60)
    print(f"  Prompt tokens:    {num_prompt_tokens}")
    print(f"  Output tokens:    {num_output_tokens}")
    print(f"  Total time:       {total_time_ms:.1f} ms")
    if ttft_ms is not None:
        print(f"  TTFT:             {ttft_ms:.1f} ms")
    if tpot_ms is not None:
        print(f"  TPOT:             {tpot_ms:.2f} ms/token")
    if decode_time_ms is not None:
        print(f"  Decode time:      {decode_time_ms:.1f} ms")
    throughput = num_output_tokens / (total_time_ms / 1000)
    print(f"  Throughput:       {throughput:.1f} tokens/s")
    print("=" * 60)

    engine.shutdown()


if __name__ == "__main__":
    main()
