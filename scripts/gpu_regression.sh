#!/usr/bin/env bash
# Run from a CUDA development environment. MODEL_PATH points to a local
# Qwen3.5-4B checkpoint; artifacts persist even if an individual check fails.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-4B checkpoint}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$(mktemp -d /tmp/rustinfer-gpu.XXXXXX)}"
mkdir -p "$ARTIFACT_DIR"
ARTIFACT_DIR="$(cd -- "$ARTIFACT_DIR" && pwd)"
if [[ -e "$ARTIFACT_DIR/service/results.json" ]]; then
    echo "Choose a new ARTIFACT_DIR; this directory already contains results." >&2
    exit 2
fi

# shellcheck source=scripts/lib/cuda_env.sh
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries

printf 'GPU regression artifacts: %s\n' "$ARTIFACT_DIR"
cargo build --release --locked -p infer-worker -p infer-scheduler -p infer-server \
    2>&1 | tee "$ARTIFACT_DIR/build.log"
cargo test --release --locked -p infer-backend-cuda --test graph_memory \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/graph-memory.log"
cargo test --release --locked -p infer-backend-cuda --test pool_memory \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/pool-memory.log"
cargo test --release --locked -p infer-backend-cuda --lib config::pool_failure_tests \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/pool-failures.log"
cargo test --release --locked -p infer-backend-cuda --lib \
    hd256_paged_prefill_matches_causal_mean_across_split_tiles \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/attention.log"
cargo test --release --locked -p infer-worker --test gdn_cuda \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/gdn.log"
python3 scripts/e2e_qwen35_smoke.py --model "$MODEL_PATH" \
    --output "$ARTIFACT_DIR/service" 2>&1 | tee "$ARTIFACT_DIR/service.log"
