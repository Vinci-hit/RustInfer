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

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi
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

cd "$REPO_ROOT"
FEATURE_ARGS=()
if python3 - "$CONFIG" <<'PYGGUF'
import sys, tomllib
from pathlib import Path
with open(sys.argv[1], "rb") as f:
    model = tomllib.load(f)["model"]
sys.exit(0 if Path(model).suffix.lower() == ".gguf" else 1)
PYGGUF
then
    FEATURE_ARGS=(--features cute-dsl)
    if [[ -z "${RUSTINFER_CUTE_DSL_PYTHON:-}" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
        export RUSTINFER_CUTE_DSL_PYTHON="$REPO_ROOT/.venv/bin/python"
    fi
fi
exec cargo run --release -p infer-worker --bin rustinfer-worker "${FEATURE_ARGS[@]}" -- --config "$CONFIG"
