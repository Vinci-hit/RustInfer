"""
vLLM single-request latency benchmark for Qwen3-4B.
Measures TTFT, TPOT, and total latency with token-level timing.
"""
import os
import time
import argparse

# Disable torch.compile, keep CUDA graph enabled
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "0"
os.environ["VLLM_USE_V1"] = "0"  # V1 engine forces compile, fallback to V0

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct")
    parser.add_argument("--prompt", type=str, default="请用中文详细介绍二叉树的中序遍历算法，并给出Python实现代码。")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=False,  # enable CUDA graph
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        llm.generate([args.prompt], sampling_params)
    print("Warmup done.\n")

    # Benchmark
    print("=" * 60)
    print("Benchmark run")
    print("=" * 60)

    t_start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    t_end = time.perf_counter()

    output = outputs[0]
    generated_text = output.outputs[0].text
    num_prompt_tokens = len(output.prompt_token_ids)
    num_output_tokens = len(output.outputs[0].token_ids)
    total_time_ms = (t_end - t_start) * 1000

    # Timing: approximate TTFT not available in offline mode, use total time
    ttft_ms = None
    decode_time_ms = None
    tpot_ms = total_time_ms / max(num_output_tokens, 1)
    metrics = getattr(output, 'metrics', None)
    if metrics is not None:
        ft = getattr(metrics, 'first_token_time', None)
        st = getattr(metrics, 'first_scheduled_time', None)
        fin = getattr(metrics, 'finished_time', None)
        if ft and st:
            ttft_ms = (ft - st) * 1000
        if ft and fin:
            decode_time_ms = (fin - ft) * 1000
            tpot_ms = decode_time_ms / max(num_output_tokens - 1, 1)

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


if __name__ == "__main__":
    main()
