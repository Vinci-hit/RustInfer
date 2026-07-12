#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Drop any inherited proxy so local server traffic stays off the proxy.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

CONFIG="${CONFIG:-${1:-$REPO_ROOT/rustinfer.toml}}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"

echo "═════════════════════════════════════════════════════"
echo "  RustInfer HTTP Server"
echo "═════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "═════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd "$REPO_ROOT"
exec cargo run --release -p infer-server --bin rustinfer-server -- --config "$CONFIG"
