#!/bin/bash
# Start RustInfer server (all-in-one: spawns scheduler + worker internally)
# Usage: bash bench/start_server.sh [batch_size] [port]

set -e

BATCH=${1:-8}
PORT=${2:-8014}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b

echo "=== RustInfer Server (batch=$BATCH, port=$PORT) ==="

# Cleanup
pkill -9 -f "rustinfer-server|rustinfer-scheduler|rustinfer-worker" 2>/dev/null || true
sleep 1

# Build
cargo build --release --features="cuda" 2>/dev/null
echo "[OK] Build done"

# Start server (foreground, add target/release to PATH so it finds scheduler/worker)
export PATH=$PWD/target/release:$PATH
exec ./target/release/rustinfer-server \
  --model $MODEL \
  --device cuda:0 \
  --port $PORT \
  --max-batch-tokens 8192 \
  --max-batch-seqs $BATCH \
  --max-model-len 4096 \
  --kv-cache-mode paged:16 \
  --log-level info
