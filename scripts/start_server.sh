#!/bin/bash
#
# Start RustInfer HTTP Server (foreground)
#
# Usage:
#   ./start_server.sh [config_path]
#
# Environment variables (override positional args):
#   CONFIG    Path to the shared TOML config (default: rustinfer.toml)
#
# Startup order: run AFTER start_scheduler.sh and start_worker.sh
#

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"

echo "═════════════════════════════════════════════════════"
echo "  RustInfer HTTP Server"
echo "═════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "═════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec cargo run --release -p infer-server --bin rustinfer-server -- --config "$CONFIG"
