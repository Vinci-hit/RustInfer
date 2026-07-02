#!/bin/bash
# One-shot e2e smoke: launch scheduler + worker + server for a given config,
# wait for the server to answer, send one chat completion, print it, tear down.
# Usage: e2e_smoke.sh <config.toml> <port> <prompt>
set -u
CONFIG="$1"
PORT="$2"
PROMPT="${3:-Say hello in one short sentence.}"

export CUDNN_FRONTEND_INCLUDE_DIR=/data/home/vinciiliu/vllm-bench/.venv/lib/python3.12/site-packages/include
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

# CUDA runtime libs on the linker path (mirror start_worker.sh probe).
if ! ldconfig -p 2>/dev/null | grep -q 'libcublas\.so\.12'; then
  _w="$(python -c 'import os,nvidia.cublas as c;print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)"
  [ -n "$_w" ] && export LD_LIBRARY_PATH="$_w:${LD_LIBRARY_PATH:-}"
fi

LOG=/tmp/e2e_$$
mkdir -p "$LOG"
BIN=./target/release
pids=()
cleanup() {
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  sleep 1
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT

echo "[e2e] scheduler..."
$BIN/rustinfer-scheduler --config "$CONFIG" >"$LOG/sched.log" 2>&1 &
pids+=($!)
sleep 2

echo "[e2e] worker..."
$BIN/rustinfer-worker --config "$CONFIG" >"$LOG/worker.log" 2>&1 &
pids+=($!)

echo "[e2e] server..."
$BIN/rustinfer-server --config "$CONFIG" >"$LOG/server.log" 2>&1 &
pids+=($!)

# Wait for weights to load + Ready (up to 120s).
echo "[e2e] waiting for worker Ready..."
for i in $(seq 1 120); do
  grep -q "Entering serve loop" "$LOG/worker.log" 2>/dev/null && break
  if ! kill -0 "${pids[1]}" 2>/dev/null; then echo "[e2e] WORKER DIED"; tail -30 "$LOG/worker.log"; exit 1; fi
  sleep 1
done

# Wait for the HTTP port.
echo "[e2e] waiting for server port $PORT..."
for i in $(seq 1 30); do
  curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

echo "[e2e] === worker eos line ==="
grep -i "eos\|no eos_token" "$LOG/worker.log" | head -3
echo "[e2e] === /v1/chat/completions ==="
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":64,\"temperature\":0}" \
  | tee "$LOG/resp.json"
echo
echo "[e2e] logs in $LOG"
