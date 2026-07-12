#!/usr/bin/env bash
# Full LLM e2e: exercise unary and streaming chat/completion endpoints against
# a three-process release stack, with bounded readiness and request deadlines.

set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 <config.toml> <port>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="$1"
PORT="$2"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-180}"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/target/release}"
BASE_URL="http://127.0.0.1:$PORT"

[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Invalid port: $PORT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"
for binary in rustinfer-scheduler rustinfer-worker rustinfer-server; do
    [[ -x "$BIN_DIR/$binary" ]] || { echo "Missing release binary: $BIN_DIR/$binary" >&2; exit 2; }
done

# shellcheck source=scripts/lib/cuda_env.sh
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rustinfer-e2e-full.XXXXXX")"
pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for name in scheduler worker server; do
    "$BIN_DIR/rustinfer-$name" --config "$CONFIG" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
done

deadline=$((SECONDS + READY_TIMEOUT_SECS))
ready=false
while (( SECONDS < deadline )); do
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "RustInfer process exited before readiness; logs: $LOG_DIR" >&2
            tail -n 40 "$LOG_DIR"/*.log >&2 || true
            exit 1
        }
    done
    if grep -q "Entering serve loop" "$LOG_DIR/worker.log" 2>/dev/null \
        && curl --fail --silent --show-error --max-time 2 "$BASE_URL/health" >/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo "RustInfer readiness timed out; logs: $LOG_DIR" >&2
    tail -n 40 "$LOG_DIR"/*.log >&2 || true
    exit 1
fi

post_json() {
    local endpoint="$1"
    local payload="$2"
    local output="$3"
    curl --fail-with-body --silent --show-error --max-time 120 \
        "$BASE_URL$endpoint" -H 'Content-Type: application/json' \
        --data-binary "$payload" >"$output"
}

chat_payload='{"model":"m","messages":[{"role":"user","content":"What is the capital of France? Answer briefly."}],"max_tokens":64,"temperature":0}'
completion_payload='{"model":"m","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
post_json /v1/chat/completions "$chat_payload" "$LOG_DIR/chat.json"
post_json /v1/completions "$completion_payload" "$LOG_DIR/completion.json"

python3 - "$LOG_DIR/chat.json" "$LOG_DIR/completion.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        response = json.load(handle)
    if not response.get("choices"):
        raise SystemExit(f"missing choices in {path}: {response}")
PY

stream_request() {
    local endpoint="$1"
    local payload="$2"
    local output="$3"
    curl --fail-with-body --silent --show-error --no-buffer --max-time 120 \
        "$BASE_URL$endpoint" -H 'Content-Type: application/json' \
        --data-binary "$payload" >"$output"
    grep -q '^data: \[DONE\]' "$output" || {
        echo "stream did not terminate with [DONE]: $output" >&2
        return 1
    }
    if grep -q '^event: error' "$output"; then
        echo "stream returned an error event: $output" >&2
        return 1
    fi
}

stream_request /v1/chat/completions \
    '{"model":"m","messages":[{"role":"user","content":"Name the capital of France."}],"max_tokens":64,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}' \
    "$LOG_DIR/chat-stream.sse"
stream_request /v1/completions \
    '{"model":"m","prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":true}' \
    "$LOG_DIR/completion-stream.sse"

echo "[e2e] full endpoint suite passed; logs: $LOG_DIR"
