#!/bin/bash
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

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"

echo "═════════════════════════════════════════════════════"
echo "  RustInfer Worker"
echo "═════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "═════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec cargo run --release -p infer-worker --bin rustinfer-worker -- --config "$CONFIG"
