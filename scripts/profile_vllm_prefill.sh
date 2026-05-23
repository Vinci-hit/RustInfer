#!/bin/bash
# Profile vLLM prefill kernel breakdown with nsys.
# Usage: bash scripts/profile_vllm_prefill.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT=/tmp/vllm_prefill

pkill -KILL -f 'vllm|nsys' 2>/dev/null || true
sleep 2

cd ~/vllm-test
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --output="$OUTPUT" \
    --force-overwrite=true \
    uv run python "$SCRIPT_DIR/vllm_prefill_workload.py"

echo ""
nsys stats --report cuda_gpu_kern_sum "${OUTPUT}.nsys-rep" 2>&1 | head -35
