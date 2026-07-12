#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-${1:-$REPO_ROOT/rustinfer.toml}}"
PROFILE_STEPS="${PROFILE_STEPS:-200}"
NSYS_OUTPUT="${NSYS_OUTPUT:-result/nsys_worker_steps}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi
command -v nsys >/dev/null 2>&1 || {
    echo "nsys is required for profiling" >&2
    exit 127
}
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"

# shellcheck source=scripts/lib/cuda_env.sh
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries

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
cd "$REPO_ROOT"
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
