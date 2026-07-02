#!/bin/bash
# Full e2e: launch scheduler + worker + server, then exercise all three LLM
# server paths (chat non-stream, chat stream/SSE, /v1/completions) so the
# deduplicated handlers + shared SSE loop are all covered. Tear down at exit.
# Usage: e2e_full.sh <config.toml> <port>
set -u
CONFIG="$1"
PORT="$2"

export CUDNN_FRONTEND_INCLUDE_DIR=/data/home/vinciiliu/vllm-bench/.venv/lib/python3.12/site-packages/include
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
if ! ldconfig -p 2>/dev/null | grep -q 'libcublas\.so\.12'; then
  _w="$(python -c 'import os,nvidia.cublas as c;print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)"
  [ -n "$_w" ] && export LD_LIBRARY_PATH="$_w:${LD_LIBRARY_PATH:-}"
fi

LOG=/tmp/e2e_$$; mkdir -p "$LOG"; BIN=./target/release; pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; sleep 1; for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT

$BIN/rustinfer-scheduler --config "$CONFIG" >"$LOG/sched.log" 2>&1 & pids+=($!); sleep 2
$BIN/rustinfer-worker    --config "$CONFIG" >"$LOG/worker.log" 2>&1 & pids+=($!)
$BIN/rustinfer-server    --config "$CONFIG" >"$LOG/server.log" 2>&1 & pids+=($!)

echo "[e2e] waiting for worker Ready..."
for i in $(seq 1 150); do
  grep -q "Entering serve loop" "$LOG/worker.log" 2>/dev/null && break
  if ! kill -0 "${pids[1]}" 2>/dev/null; then echo "[e2e] WORKER DIED"; tail -30 "$LOG/worker.log"; exit 1; fi
  sleep 1
done
for i in $(seq 1 30); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

B="http://127.0.0.1:$PORT"
Q='What is the capital of France? Answer in one short sentence.'

echo "[e2e] === 1) chat non-stream ==="
curl -s "$B/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":64,\"temperature\":0}"
echo
echo "[e2e] === 2) chat STREAM (SSE, last 6 lines) ==="
curl -s -N "$B/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":64,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  | tail -6
echo
echo "[e2e] === 3) /v1/completions non-stream ==="
curl -s "$B/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"
echo
echo "[e2e] === 4) /v1/completions STREAM (last 4 lines) ==="
curl -s -N "$B/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0,\"stream\":true}" \
  | tail -4
echo
echo "[e2e] logs in $LOG"
