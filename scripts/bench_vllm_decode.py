"""vLLM 多 batch decode 吞吐 benchmark。

跟 RustInfer `perf_<model>_decode_matrix` 同协议：
- batch ∈ {1, 2, 4, 8}
- 每条 seq 用相同 prompt，强制 decode 满 `--decode-steps`（min/max tokens 同值）
- 计时 = `vllm.LLM.generate(prompts, sampling)` 的 wall time
- 报告 tokens/s = (batch * decode_steps) / wall_time
- 默认 `enforce_eager=False` 让 vLLM 启用自家 CUDA Graph

输出表格与 Rust 端对齐，便于人工对照。
"""
import os
import time
import argparse

# Disable torch.compile, keep CUDA graph enabled
os.environ.setdefault("VLLM_TORCH_COMPILE_LEVEL", "0")
os.environ.setdefault("VLLM_USE_V1", "0")

# 容器里 NVML 不可用（`Can't initialize NVML` warning），torch 仍能正常跑 CUDA，
# 但 vLLM 0.20+ 的 platform 自动检测依赖 `pynvml.nvmlDeviceGetCount() > 0`，
# NVML init 失败就 fallback 到 CpuPlatform，最终 device_type 为空 → 报
# `RuntimeError: Device string must not be empty`。
#
# 这里在 import vllm 之前 monkey-patch 让 cuda_platform_plugin 直接返回 cuda
# qualname（前提是 torch.cuda.is_available() 为真，避免在真 CPU 机器上误判）。
import torch as _torch
if _torch.cuda.is_available():
    import vllm.platforms as _vp

    def _force_cuda_plugin():
        return "vllm.platforms.cuda.CudaPlatform"

    _vp.cuda_platform_plugin = _force_cuda_plugin
    _vp.builtin_platform_plugins["cuda"] = _force_cuda_plugin

from vllm import LLM, SamplingParams


def bench_one(llm: LLM, prompt: str, batch: int, decode_steps: int) -> dict:
    sampling = SamplingParams(
        temperature=0.0,             # greedy 与 RustInfer 对齐（其默认 sampler 也是 argmax）
        max_tokens=decode_steps,
        min_tokens=decode_steps,     # 强制满 N 步，避免 EOS 提前
        ignore_eos=True,
    )
    prompts = [prompt] * batch

    # Warmup 一次（同 batch_size，让 vLLM 自己 graph capture）
    llm.generate(prompts, sampling, use_tqdm=False)

    # 计时
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    t1 = time.perf_counter()
    wall_s = t1 - t0

    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    expected = batch * decode_steps
    if total_out_tokens != expected:
        # 极少数情况 vLLM 会少输出（早停 / 截断），按实际算
        pass
    tokens_per_sec = total_out_tokens / wall_s
    return {
        "batch": batch,
        "wall_ms": wall_s * 1000.0,
        "out_tokens": total_out_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="vllm", help="header label，e.g. 'Llama3-1B'")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--batches", default="1,2,4,8", help="逗号分隔 batch sizes")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem-util", type=float, default=0.5)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    max_batch = max(batches)

    print(f"Loading vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=max_batch,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
    )

    print()
    print("══════════════════════════════════════════════════════════════════")
    print(f"  {args.label} (vLLM) —— decode benchmark (steps={args.decode_steps})")
    print("══════════════════════════════════════════════════════════════════")
    print(f"{'batch':>6}  {'wall ms':>10}  {'out tokens':>11}  {'tokens/s':>12}")
    print(f"{'─────':>6}  {'──────────':>10}  {'───────────':>11}  {'────────────':>12}")
    for bs in batches:
        r = bench_one(llm, args.prompt, bs, args.decode_steps)
        print(f"{r['batch']:>6}  {r['wall_ms']:>10.1f}  {r['out_tokens']:>11}  {r['tokens_per_sec']:>12.1f}")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
