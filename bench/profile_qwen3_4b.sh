#!/bin/bash
# Profile RustInfer vs vLLM with Qwen3-4B
# Usage: bash bench/profile_qwen3_4b.sh [ri|vllm|both] [duration]

set -e

MODE=${1:-both}
DURATION=${2:-10}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct
RI_PORT=8014
VLLM_PORT=8001
BATCH=32
RESULT_DIR=result
VLLM_PY=${VLLM_PY:-/root/vllm-bench/bin/python}

# Readiness checks below use curl against localhost. A corporate http_proxy in the
# environment otherwise routes these through the proxy, which returns HTTP 502 for a
# not-yet-listening port — making curl exit 0 and falsely report "Ready after 1s".
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

mkdir -p $RESULT_DIR

echo "=== Profile Qwen3-4B: $MODE, duration=${DURATION}s ==="
echo ""

profile_ri() {
    echo "[RI] Starting RustInfer under nsys..."
    pkill -9 -f "rustinfer-server|rustinfer-scheduler|rustinfer-worker" 2>/dev/null || true
    sleep 1

    cargo build --release --features="cuda" 2>/dev/null
    export PATH=$PWD/target/release:$PATH

    nsys profile \
      --trace=cuda,cublas \
      --cuda-graph-trace=node \
      --trace-fork-before-exec=true \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      --output=$RESULT_DIR/nsys_qwen3_4b_ri \
      ./target/release/rustinfer-server \
        --model $MODEL \
        --device cuda:0 \
        --port $RI_PORT \
        --max-batch-tokens 8192 \
        --max-batch-seqs $BATCH \
        --max-model-len 4096 \
        --kv-cache-mode paged:16 \
        --log-level warn &
    RI_PID=$!

    echo "  PID=$RI_PID, waiting for model load..."
    for i in $(seq 1 120); do
      if ! kill -0 $RI_PID 2>/dev/null; then
        echo "  ERROR: Server died"
        return 1
      fi
      if curl -s http://localhost:$RI_PORT/v1/models >/dev/null 2>&1; then
        echo "  Ready after ${i}s"
        break
      fi
      sleep 1
    done

    echo "[RI] Running benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
    python3 bench/bench_online_burst.py \
      --target rustinfer --port $RI_PORT \
      --duration $DURATION --concurrency $BATCH \
      --output /tmp/bench_profile_qwen3_ri.json 2>&1 | tail -15

    echo "[RI] Stopping server..."
    kill -INT $RI_PID 2>/dev/null || true
    sleep 3
    kill -9 $RI_PID 2>/dev/null || true
    wait $RI_PID 2>/dev/null || true
    sleep 2

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  nsys kernel stats: RustInfer Qwen3-4B"
    echo "═══════════════════════════════════════════════════════"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format table $RESULT_DIR/nsys_qwen3_4b_ri.nsys-rep 2>&1 | head -50
    echo ""
}

profile_vllm() {
    echo "[vLLM] Starting vLLM under nsys..."
    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep 1

    nsys profile \
      --trace=cuda,cublas \
      --cuda-graph-trace=node \
      --trace-fork-before-exec=true \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      --output=$RESULT_DIR/nsys_qwen3_4b_vllm \
      "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $VLLM_PORT \
        --max-model-len 4096 \
        --max-num-seqs $BATCH \
        --gpu-memory-utilization 0.9 > $RESULT_DIR/vllm_qwen3_startup.log 2>&1 &
    VLLM_PID=$!

    echo "  PID=$VLLM_PID, waiting for model load..."
    for i in $(seq 1 120); do
      if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM died. Log:"
        tail -20 $RESULT_DIR/vllm_qwen3_startup.log
        return 1
      fi
      if curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then
        echo "  Ready after ${i}s"
        break
      fi
      sleep 1
    done

    echo "[vLLM] Running benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
    python3 bench/bench_online_burst.py \
      --target vllm --port $VLLM_PORT \
      --duration $DURATION --concurrency $BATCH \
      --output /tmp/bench_profile_qwen3_vllm.json 2>&1 | tail -15

    echo "[vLLM] Stopping server..."
    kill -INT $VLLM_PID 2>/dev/null || true
    sleep 5
    kill -9 $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 2

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  nsys kernel stats: vLLM Qwen3-4B"
    echo "═══════════════════════════════════════════════════════"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format table $RESULT_DIR/nsys_qwen3_4b_vllm.nsys-rep 2>&1 | head -50
    echo ""
}

case $MODE in
  ri)   profile_ri ;;
  vllm) profile_vllm ;;
  both)
    profile_ri
    echo ""
    echo "════════════════════════════════════════"
    echo ""
    profile_vllm
    ;;
esac

echo "Done. Reports:"
echo "  RI:   $RESULT_DIR/nsys_qwen3_4b_ri.nsys-rep"
echo "  vLLM: $RESULT_DIR/nsys_qwen3_4b_vllm.nsys-rep"
