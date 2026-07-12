#!/usr/bin/env bash
# One-shot LLM smoke test: launch the three release binaries, wait for bounded
# readiness, issue one chat completion, validate it, and tear the stack down.

set -euo pipefail

usage() {
    echo "Usage: $0 <config.toml> <port> [prompt]" >&2
}

if (( $# < 2 || $# > 3 )); then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="$1"
PORT="$2"
PROMPT="${3:-Say hello in one short sentence.}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-180}"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/target/release}"

[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Invalid port: $PORT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd)/$(basename -- "$CONFIG")"
for binary in rustinfer-scheduler rustinfer-worker rustinfer-server; do
    [[ -x "$BIN_DIR/$binary" ]] || {
        echo "Missing release binary: $BIN_DIR/$binary" >&2
        echo "Build first with: cargo build --release" >&2
        exit 2
    }
done

# shellcheck source=scripts/lib/cuda_env.sh
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rustinfer-e2e-smoke.XXXXXX")"
pids=()
cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

start_process() {
    local name="$1"
    "$BIN_DIR/rustinfer-$name" --config "$CONFIG" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
}

start_process scheduler
start_process worker
start_process server

deadline=$((SECONDS + READY_TIMEOUT_SECS))
ready=false
while (( SECONDS < deadline )); do
    for index in "${!pids[@]}"; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null; then
            echo "A RustInfer process exited before readiness; logs: $LOG_DIR" >&2
            tail -n 40 "$LOG_DIR"/*.log >&2 || true
            exit 1
        fi
    done
    if grep -q "Entering serve loop" "$LOG_DIR/worker.log" 2>/dev/null \
        && curl --fail --silent --show-error --max-time 2 \
            "http://127.0.0.1:$PORT/health" >/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo "RustInfer did not become ready within ${READY_TIMEOUT_SECS}s; logs: $LOG_DIR" >&2
    tail -n 40 "$LOG_DIR"/*.log >&2 || true
    exit 1
fi

payload="$(python3 - "$PROMPT" <<'PY'
import json
import sys

print(json.dumps({
    "model": "m",
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": 64,
    "temperature": 0,
}))
PY
)"
response="$LOG_DIR/response.json"
curl --fail-with-body --silent --show-error --max-time 120 \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "$payload" >"$response"

python3 - "$response" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
choices = response.get("choices") or []
if not choices or not choices[0].get("message", {}).get("content"):
    raise SystemExit(f"invalid completion response: {response}")
print(json.dumps(response, ensure_ascii=False))
PY

echo "[e2e] smoke passed; logs: $LOG_DIR"
