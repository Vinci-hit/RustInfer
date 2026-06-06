#!/bin/bash
# Start RustInfer server (launches the 3 split processes: scheduler + worker + api)
# Usage: bash bench/start_server.sh [config_path]
#
# Env override:
#   CONFIG   Path to the shared TOML config (default: rustinfer.toml)

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"

echo "=== RustInfer Server (config=$CONFIG) ==="

# Cleanup
pkill -9 -f "rustinfer-scheduler|rustinfer-worker|rustinfer-server" 2>/dev/null || true
sleep 1

# Build
cargo build --release --features="cuda" 2>/dev/null
echo "[OK] Build done"

BIN=$PWD/target/release

# Tear down all children on exit (Ctrl+C).
PIDS=()
cleanup() {
    echo ""
    echo "Shutting down (scheduler/worker/api)..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# 1. Scheduler (binds IPC sockets first)
echo "[1/3] Starting scheduler..."
"$BIN/rustinfer-scheduler" --config "$CONFIG" &
PIDS+=($!)
sleep 2

# 2. Worker
echo "[2/3] Starting worker..."
"$BIN/rustinfer-worker" --config "$CONFIG" &
PIDS+=($!)
sleep 2

# 3. HTTP server (foreground via wait)
echo "[3/3] Starting HTTP server..."
"$BIN/rustinfer-server" --config "$CONFIG" &
PIDS+=($!)

echo "All processes started. Press Ctrl+C to stop."
wait
