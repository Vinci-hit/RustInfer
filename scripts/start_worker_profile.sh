#!/usr/bin/env bash
#
# Start RustInfer Worker (foreground)
#
# Usage:
#   ./start_worker.sh [config_path]
#
# Environment variables (override positional args):
#   CONFIG    Path to the shared TOML config (default: rustinfer.toml)
#
# Startup order: run AFTER start_scheduler.sh, then run start_api.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-${1:-$REPO_ROOT/rustinfer.toml}}"
DELAY="${DELAY:-30}"
DURATION="${DURATION:-10}"
NSYS_OUTPUT="${NSYS_OUTPUT:-result/nsys_worker}"

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

echo "═════════════════════════════════════════════════════"
echo "  RustInfer Worker"
echo "═════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "═════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""
mkdir -p -- "$(dirname -- "$NSYS_OUTPUT")"
cd "$REPO_ROOT"
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-graph-trace=node \
    --output="$NSYS_OUTPUT" \
    --export=sqlite \
    --delay="$DELAY" \
    --duration="$DURATION" \
    --kill=sigkill \
    --force-overwrite=true \
    cargo run --release -p infer-worker --bin rustinfer-worker -- --config "$CONFIG"
