#!/bin/bash
# Profile vLLM with nsys
# Usage: bash bench/profile_vllm.sh [batch_size] [port] [duration]

set -e

BATCH=${1:-8}
PORT=${2:-8001}
DURATION=${3:-5}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b
RESULT_DIR=result
RESULT_FILE=$RESULT_DIR/nsys_vllm_batch${BATCH}
VLLM_PY=${VLLM_PY:-/root/vllm-bench/bin/python}

# Readiness checks use curl against localhost; bypass any corporate http_proxy so a
# not-yet-listening port returns a connection error (curl exit!=0) instead of a 502.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

echo "=== Profile vLLM batch=$BATCH, port=$PORT, duration=${DURATION}s ==="

# Cleanup (kill python vllm processes, not this script)
pkill -9 -f "python.*vllm" 2>/dev/null || true
sleep 1
mkdir -p $RESULT_DIR

# Start vLLM under nsys (log to file so we can see errors)
VLLM_LOG=$RESULT_DIR/vllm_startup.log
echo "[1/3] Starting vLLM under nsys (log: $VLLM_LOG)..."
nsys profile \
  --trace=cuda,cublas \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=$RESULT_FILE \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --max-model-len 4096 \
    --max-num-seqs $BATCH \
    --enforce-eager false \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats > $VLLM_LOG 2>&1 &
VLLM_PID=$!

echo "  vLLM PID=$VLLM_PID, waiting for model load..."
# Wait for vLLM to be ready (up to 120s)
READY=0
for i in $(seq 1 120); do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "  ERROR: vLLM process died. Log:"
    tail -30 $VLLM_LOG
    exit 1
  fi
  if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
    echo "  vLLM ready after ${i}s"
    READY=1
    break
  fi
  sleep 1
done

if [ $READY -eq 0 ]; then
  echo "  ERROR: vLLM did not become ready in 120s. Log:"
  tail -30 $VLLM_LOG
  kill -9 $VLLM_PID 2>/dev/null || true
  exit 1
fi

# Run benchmark
echo "[2/3] Running burst benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
python3 bench/bench_online_burst.py \
  --target vllm --port $PORT \
  --duration $DURATION --concurrency $BATCH \
  --output /tmp/bench_profile_vllm_batch${BATCH}.json 2>&1 | tail -20

# Stop vLLM
echo ""
echo "[3/3] Stopping vLLM, collecting nsys data..."
kill -INT $VLLM_PID 2>/dev/null || true
sleep 5
kill -9 $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
sleep 2

# Generate stats
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  nsys kernel stats: vLLM batch=$BATCH"
echo "═══════════════════════════════════════════════════════"
nsys stats --force-export=true --report cuda_gpu_kern_sum --format table $RESULT_FILE.nsys-rep 2>&1 | head -50

echo ""
echo "Full report: $RESULT_FILE.nsys-rep"
echo "View: nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $RESULT_FILE.nsys-rep"
