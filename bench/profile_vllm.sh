#!/bin/bash
# vLLM startup script for Qwen3-4B-Instruct
# Usage: bash start_vllm.sh [extra_args...]

set -euo pipefail

# === Model Config ===
MODEL_PATH="/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct"

# === Runtime Config ===
MAX_NUM_SEQS=32
MAX_MODEL_LEN=1024
MAX_NUM_BATCHED_TOKENS=2048

# === GPU Memory ===
GPU_MEMORY_UTIL=0.87

# === Prefix Caching ===
ENABLE_PREFIX_CACHING=false

# === Tensor Parallel ===
TENSOR_PARALLEL_SIZE=1

# === Port ===
PORT=8000

# === Nsight Systems Config ===
DELAY=30
DURATION=10
NSYS_OUTPUT="vllm_qwen_analysis"

VLLM_ARGS=(
    "$MODEL_PATH"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --block-size 16
)

if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    VLLM_ARGS+=(--enable-prefix-caching)
else
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --output="$NSYS_OUTPUT" \
    --export=sqlite \
    --delay="$DELAY" \
    --duration="$DURATION" \
    --kill=sigkill \
    --force-overwrite=true \
    uv run vllm serve \
    "${VLLM_ARGS[@]}" \
    "$@"
