#!/bin/bash
# vLLM startup script for Qwen3-4B-Instruct
# Based on bootstrap log configuration
# Usage: bash start_vllm.sh [extra_args...]

set -euo pipefail

# === Model Config ===
MODEL_PATH="/root/models/Qwen3-4B-Instruct-2507"

# === Runtime Config (from bootstrap log) ===
MAX_NUM_SEQS=32
MAX_MODEL_LEN=1024
MAX_NUM_BATCHED_TOKENS=2048

# === GPU Memory ===
# Total: 95GB, Free: 83GB → use ~0.87 to leave headroom
GPU_MEMORY_UTIL=0.87

# === Prefix Caching (disabled) ===
ENABLE_PREFIX_CACHING=false

# === Tensor Parallel (single GPU) ===
TENSOR_PARALLEL_SIZE=1

# === Port ===
PORT=8000

# === Build vllm serve command ===
exec uv run vllm serve \
    "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --no-enable-prefix-caching \
    --block-size 16 \
    "$@"
