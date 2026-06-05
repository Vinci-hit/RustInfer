#!/bin/bash
# Bench RustInfer vs vLLM with Qwen3-4B
# Usage: bash bench/bench_qwen3_4b.sh [ri|vllm|both]

set -e

MODE=${1:-both}
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct
RI_PORT=8014
VLLM_PORT=8001
BATCH=32
DURATION=60
VLLM_PY=${VLLM_PY:-/root/vllm-bench/bin/python}

# Readiness checks use curl against localhost; bypass any corporate http_proxy so a
# not-yet-listening port returns a connection error (curl exit!=0) instead of a 502.
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

echo "=== Qwen3-4B Benchmark: $MODE ==="
echo ""

run_ri() {
    echo "[RI] Starting RustInfer server..."
    pkill -9 -f "rustinfer-server|rustinfer-scheduler|rustinfer-worker" 2>/dev/null || true
    sleep 1

    cargo build --release --features="cuda" 2>/dev/null
    export PATH=$PWD/target/release:$PATH

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

    echo "[RI] Waiting for server (PID=$RI_PID)..."
    for i in $(seq 1 120); do
      if ! kill -0 $RI_PID 2>/dev/null; then
        echo "[RI] ERROR: Server died"
        return 1
      fi
      if curl -s http://localhost:$RI_PORT/v1/models >/dev/null 2>&1; then
        echo "[RI] Ready after ${i}s"
        break
      fi
      sleep 1
    done

    echo "[RI] Running benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
    python3 bench/bench_online_burst.py \
      --target rustinfer --port $RI_PORT \
      --duration $DURATION --concurrency $BATCH \
      --output /tmp/bench_qwen3_4b_ri.json

    kill $RI_PID 2>/dev/null || true
    wait $RI_PID 2>/dev/null || true
    echo "[RI] Done."
}

run_vllm() {
    echo "[vLLM] Starting vLLM server..."
    pkill -9 -f "python.*vllm" 2>/dev/null || true
    sleep 1

    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model $MODEL \
      --port $VLLM_PORT \
      --max-model-len 4096 \
      --max-num-seqs $BATCH \
      --gpu-memory-utilization 0.9 &
    VLLM_PID=$!

    echo "[vLLM] Waiting for server (PID=$VLLM_PID)..."
    for i in $(seq 1 120); do
      if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[vLLM] ERROR: Server died"
        return 1
      fi
      if curl -s http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then
        echo "[vLLM] Ready after ${i}s"
        break
      fi
      sleep 1
    done

    echo "[vLLM] Running benchmark (concurrency=$BATCH, duration=${DURATION}s)..."
    python3 bench/bench_online_burst.py \
      --target vllm --port $VLLM_PORT \
      --duration $DURATION --concurrency $BATCH \
      --output /tmp/bench_qwen3_4b_vllm.json

    kill $VLLM_PID 2>/dev/null || true
    sleep 3
    kill -9 $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "[vLLM] Done."
}

case $MODE in
  ri)   run_ri ;;
  vllm) run_vllm ;;
  both)
    run_ri
    echo ""
    echo "════════════════════════════════════════"
    echo ""
    run_vllm
    ;;
esac

# Compare results
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Results Summary: Qwen3-4B (batch=$BATCH, ${DURATION}s)"
echo "═══════════════════════════════════════════════════════"
python3 -c "
import json
for label, path in [('RustInfer', '/tmp/bench_qwen3_4b_ri.json'), ('vLLM', '/tmp/bench_qwen3_4b_vllm.json')]:
    try:
        with open(path) as f:
            s = json.load(f)['stats']
        print(f'  {label:10s}: {s[\"throughput_tok_s\"]:7.0f} tok/s | p50={s[\"latency_p50\"]:.3f}s p90={s[\"latency_p90\"]:.3f}s | {s[\"successful\"]} reqs')
    except:
        print(f'  {label:10s}: (no data)')
" 2>/dev/null || true
