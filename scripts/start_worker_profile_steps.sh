#!/bin/bash
#
# Profile RustInfer Worker with --profile-cuda-steps (cudaProfilerApi range)
#
# Usage:
#   ./start_worker_profile_steps.sh [config_path]
#
# Environment variables (override positional args):
#   CONFIG            Path to the shared TOML config (default: rustinfer.toml)
#   PROFILE_STEPS     Number of worker steps to profile (default: 200)
#   NSYS_OUTPUT       Output file path for nsys report (default: result/nsys_worker_steps)
#
# Startup order: run AFTER start_scheduler.sh
#

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"
PROFILE_STEPS="${PROFILE_STEPS:-200}"
NSYS_OUTPUT="${NSYS_OUTPUT:-result/nsys_worker_steps}"

mkdir -p "$(dirname "$NSYS_OUTPUT")"

echo "═════════════════════════════════════════════════════"
echo "  RustInfer Worker (Nsight Profile - cudaProfilerApi)"
echo "═════════════════════════════════════════════════════"
echo "  Config:        $CONFIG"
echo "  Profile steps: $PROFILE_STEPS"
echo "  Nsys output:   $NSYS_OUTPUT"
echo "═════════════════════════════════════════════════════"
echo ""

# Build first (release mode)
cargo build --release -p infer-worker --bin rustinfer-worker

nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --cuda-trace-all-apis=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --sample=none \
    --cpuctxsw=none \
    --kill=none \
    --force-overwrite=true \
    --output="$NSYS_OUTPUT" \
    ./target/release/rustinfer-worker \
        --config "$CONFIG" \
        --profile-cuda-steps "$PROFILE_STEPS"
