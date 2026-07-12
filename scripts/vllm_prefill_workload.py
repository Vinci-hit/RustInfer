"""Offline vLLM prefill profiling workload."""
import os

from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH")
if not MODEL_PATH:
    raise SystemExit("Set MODEL_PATH to a local model path or model ID")

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=8192,
    max_num_seqs=32,
    gpu_memory_utilization=0.5,
)

# Warmup
llm.generate(["hello world"], SamplingParams(max_tokens=1, temperature=0))
print("WARMUP DONE", flush=True)

# Prefill workload: 10 long prompts
prompts = [
    "Write a comprehensive essay about transformer architecture including "
    "self-attention multi-head attention positional encoding feed-forward "
    "networks layer normalization residual connections and how all these "
    "components work together in modern large language models like GPT BERT "
    "T5 and their variants."
] * 10

results = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0))
print(
    f"Done: {len(results)} results, "
    f"tokens: {sum(len(r.outputs[0].token_ids) for r in results)}",
    flush=True,
)
