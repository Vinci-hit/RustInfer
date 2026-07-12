#!/usr/bin/env bash
#
# Start RustInfer Scheduler (foreground)
#
# Usage:
#   ./start_scheduler.sh [config_path]
#
# Environment variables (override positional args):
#   CONFIG    Path to the shared TOML config (default: rustinfer.toml)
#
# Startup order: run this FIRST, then start_worker.sh, then start_api.sh
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

echo "══════════════════════════════════════════════════════"
echo "  RustInfer Scheduler"
echo "══════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$REPO_ROOT"
exec cargo run --release -p infer-scheduler -- --config "$CONFIG"
