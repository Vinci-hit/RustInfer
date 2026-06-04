#!/bin/bash
# Profile RustInfer with nsys (wraps server, captures worker subprocess CUDA)
# Usage: bash bench/profile_server.sh [batch_size] [port] [duration]

set -e

BATCH=${1:-8}
PORT=${2:-8014}
DURATION=${3:-30}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b
RESULT_DIR=result
RESULT_FILE=$RESULT_DIR/nsys_batch${BATCH}

echo "=== Profile RustInfer batch=$BATCH, port=$PORT, duration=${DURATION}s ==="

# Cleanup
pkill -9 -f "rustinfer-server|rustinfer-scheduler|rustinfer-worker" 2>/dev/null || true
sleep 1
mkdir -p $RESULT_DIR

# Build
cargo build --release --features="cuda" 2>/dev/null
echo "[OK] Build done"

export PATH=$PWD/target/release:$PATH

# Start server under nsys (captures all child processes)
echo "[1/3] Starting server under nsys..."
nsys profile \
  --trace=cuda,cublas \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=$RESULT_FILE \
  ./target/release/rustinfer-server \
    --model $MODEL \
    --device cuda:0 \
    --port $PORT \
    --max-batch-tokens 8192 \
    --max-batch-seqs $BATCH \
    --max-model-len 4096 \
    --kv-cache-mode paged:16 \
    --log-level warn &
SERVER_PID=$!

echo "  Server PID=$SERVER_PID, waiting for model load..."
sleep 20

# Run benchmark
echo "[2/3] Running burst benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
python3 bench/bench_online_burst.py \
  --target rustinfer --port $PORT \
  --duration $DURATION --concurrency $BATCH \
  --output /tmp/bench_profile_batch${BATCH}.json 2>&1 | tail -20

# Stop server (nsys finishes capture)
echo ""
echo "[3/3] Stopping server, collecting nsys data..."
kill -INT $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

# Generate stats
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  nsys kernel stats: batch=$BATCH"
echo "═══════════════════════════════════════════════════════"
nsys stats --force-export=true --report cuda_gpu_kern_sum --format table $RESULT_FILE.nsys-rep 2>&1 | head -50

echo ""
echo "Full report: $RESULT_FILE.nsys-rep"
echo "View: nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $RESULT_FILE.nsys-rep"
