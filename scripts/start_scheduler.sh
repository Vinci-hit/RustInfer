#!/bin/bash
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

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"

echo "══════════════════════════════════════════════════════"
echo "  RustInfer Scheduler"
echo "══════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec cargo run --release -p infer-scheduler -- --config "$CONFIG"
