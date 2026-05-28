#!/bin/bash
# Run benchmark against running RustInfer server
# Usage: bash bench/run_bench.sh [concurrency] [duration] [port]

CONCURRENCY=${1:-8}
DURATION=${2:-30}
PORT=${3:-8014}

echo "=== Bench: concurrency=$CONCURRENCY, duration=${DURATION}s, port=$PORT ==="

python3 bench/bench_online_burst.py \
  --target rustinfer --port $PORT \
  --duration $DURATION --concurrency $CONCURRENCY \
  --output /tmp/bench_burst_c${CONCURRENCY}.json

echo ""
echo "Result saved: /tmp/bench_burst_c${CONCURRENCY}.json"
