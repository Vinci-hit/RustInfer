#!/bin/bash
# Profile RustInfer worker with nsys (batch=8 decode focus)
# Usage: bash bench/profile_batch.sh [batch_size] [duration]
#
# One-shot script: starts scheduler, nsys-profiled worker, server,
# runs burst benchmark, then collects nsys stats.

set -e

BATCH=${1:-8}
DURATION=${2:-30}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b
RESULT_DIR=result
RESULT_FILE=$RESULT_DIR/nsys_batch${BATCH}
PORT=8014

echo "=== Profile RustInfer batch=$BATCH, duration=${DURATION}s ==="
echo ""

# Cleanup
pkill -9 -f "rustinfer-scheduler|rustinfer-worker|rustinfer-server" 2>/dev/null || true
rm -f /tmp/rustinfer-nsys-*.ipc
sleep 2
mkdir -p $RESULT_DIR

# Build
echo "[1/5] Building..."
cargo build --release --features="cuda" 2>/dev/null
echo "  Done."

# Start scheduler (background)
echo "[2/5] Starting scheduler (max_batch_seqs=$BATCH)..."
PATH=$PWD/target/release:$PATH \
./target/release/rustinfer-scheduler \
  --frontend-endpoint ipc:///tmp/rustinfer-nsys-frontend.ipc \
  --worker-push-endpoint ipc:///tmp/rustinfer-nsys-worker-in.ipc \
  --worker-pull-endpoint ipc:///tmp/rustinfer-nsys-worker-out.ipc \
  --worker-control-endpoint ipc:///tmp/rustinfer-nsys-worker-control.ipc \
  --model $MODEL \
  --model-type llama3 \
  --device cuda:0 \
  --max-batch-tokens 8192 \
  --max-batch-seqs $BATCH \
  --max-model-len 4096 \
  --kv-cache-mode paged:16 \
  --log-level warn &>/dev/null &
SCHED_PID=$!
sleep 1

# Start worker under nsys with cudaProfilerApi capture
echo "[3/5] Starting worker under nsys (profile-cuda-steps=200)..."
nsys profile \
  --trace=cuda,cublas \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=$RESULT_FILE \
  ./target/release/rustinfer-worker \
    --device cuda:0 \
    --worker-pull-endpoint ipc:///tmp/rustinfer-nsys-worker-in.ipc \
    --worker-push-endpoint ipc:///tmp/rustinfer-nsys-worker-out.ipc \
    --worker-control-endpoint ipc:///tmp/rustinfer-nsys-worker-control.ipc \
    --max-batch-tokens 8192 \
    --max-batch-seqs $BATCH \
    --profile-cuda-steps 200 \
    --log-level warn &
WORKER_PID=$!
sleep 15  # wait for model load + graph priming

# Start server (background)
echo "[4/5] Starting server on port $PORT..."
PATH=$PWD/target/release:$PATH \
./target/release/rustinfer-server \
  --port $PORT \
  --engine-endpoint ipc:///tmp/rustinfer-nsys-frontend.ipc \
  --tokenizer $MODEL \
  --model-name llama3.2-1b \
  --log-level warn &>/dev/null &
SERVER_PID=$!
sleep 2

# Run benchmark
echo "[5/5] Running burst benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
python3 bench/bench_online_burst.py \
  --target rustinfer --port $PORT \
  --duration $DURATION --concurrency $BATCH \
  --output /tmp/bench_profile_batch${BATCH}.json 2>&1 | tail -15

# Wait for nsys duration to complete
echo ""
echo "Waiting for nsys capture to finish (30s)..."
sleep 35
# nsys will have finished capture, kill worker
kill $WORKER_PID 2>/dev/null || true
wait $WORKER_PID 2>/dev/null || true

# Cleanup processes
kill $SERVER_PID $SCHED_PID 2>/dev/null || true
sleep 1

# Generate stats
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  nsys kernel stats: batch=$BATCH"
echo "═══════════════════════════════════════════════════════"
nsys stats --force-export=true --report cuda_gpu_kern_sum --format table $RESULT_FILE.nsys-rep 2>&1 | head -30

echo ""
echo "Full report: $RESULT_FILE.nsys-rep"
echo "View timeline: nsys-ui $RESULT_FILE.nsys-rep"
echo ""
echo "For CSV export:"
echo "  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $RESULT_FILE.nsys-rep"
