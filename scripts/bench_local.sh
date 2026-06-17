#!/bin/bash
# Local perf bench harness: launch scheduler+worker+server on GPU 7 against
# the local Llama-3.2-1B model, wait for readiness, run the online bench, tear
# down. Usage: bash scripts/bench_local.sh [duration_s] [concurrency] [label]
set -u

REPO=/mnt/md2/liuwenqi/RustInfer
CFG=$REPO/rustinfer.bench.toml
DUR="${1:-30}"
CONC="${2:-32}"
LABEL="${3:-run}"
LOGDIR=/tmp/rustinfer_bench
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=/home/liuwenqi/miniconda3/lib:${LD_LIBRARY_PATH:-}

cleanup() {
  pkill -f "rustinfer-server --config $CFG" 2>/dev/null
  pkill -f "rustinfer-worker --config $CFG" 2>/dev/null
  pkill -f "infer-scheduler -- --config $CFG" 2>/dev/null
  pkill -f "rustinfer-scheduler --config $CFG" 2>/dev/null
  sleep 1
}
trap cleanup EXIT
cleanup

cd "$REPO"
echo "[harness] launching scheduler..."
./target/release/rustinfer-scheduler --config "$CFG" >"$LOGDIR/sched.log" 2>&1 &
sleep 2
echo "[harness] launching worker on GPU 7..."
./target/release/rustinfer-worker --config "$CFG" >"$LOGDIR/worker.log" 2>&1 &
echo "[harness] launching server..."
./target/release/rustinfer-server --config "$CFG" >"$LOGDIR/server.log" 2>&1 &

echo "[harness] waiting for worker Ready + :8000 ..."
for i in $(seq 1 120); do
  if grep -q "Entering serve loop" "$LOGDIR/worker.log" 2>/dev/null && \
     curl -s -o /dev/null http://127.0.0.1:8000/v1/models 2>/dev/null; then
    echo "[harness] ready after ${i}s"
    break
  fi
  sleep 1
done

if ! curl -s -o /dev/null http://127.0.0.1:8000/v1/models 2>/dev/null; then
  echo "[harness] SERVER NOT READY — tail worker.log:"; tail -20 "$LOGDIR/worker.log"
  exit 1
fi

echo "[harness] running bench: dur=${DUR}s conc=${CONC} label=${LABEL}"
python3 bench/_local_compare.py --tag rustinfer --duration "$DUR" --concurrency "$CONC" 2>&1
cp /tmp/bench_online_rustinfer.json "$LOGDIR/bench_${LABEL}.json" 2>/dev/null
echo "[harness] saved $LOGDIR/bench_${LABEL}.json"
