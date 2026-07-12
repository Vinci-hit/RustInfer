#!/usr/bin/env bash
# Local RustInfer online benchmark harness.
# Environment: CONFIG, PORT, BIN_DIR, READY_TIMEOUT_SECS may override defaults.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/rustinfer.toml}"
PORT="${PORT:-8000}"
BIN_DIR="${BIN_DIR:-$REPO_ROOT/target/release}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-180}"
DURATION="${1:-30}"
CONCURRENCY="${2:-32}"
LABEL="${3:-run}"
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/rustinfer-bench}"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
mkdir -p "$LOG_DIR"

# shellcheck source=scripts/lib/cuda_env.sh
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries

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
            echo "RustInfer process exited before readiness" >&2
            tail -n 40 "$LOG_DIR"/*.log >&2 || true
            exit 1
        }
    done
    if grep -q "Entering serve loop" "$LOG_DIR/worker.log" 2>/dev/null \
        && curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo "Benchmark stack readiness timed out" >&2; exit 1; }

python3 "$REPO_ROOT/bench/_local_compare.py" \
    --tag rustinfer \
    --url "http://127.0.0.1:$PORT" \
    --duration "$DURATION" \
    --concurrency "$CONCURRENCY"

cp "/tmp/bench_online_rustinfer.json" "$LOG_DIR/bench_${LABEL}.json"
echo "[bench] saved $LOG_DIR/bench_${LABEL}.json"
