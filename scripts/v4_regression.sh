#!/usr/bin/env bash
# Model-free V4 kernel regression, also used by the trusted GPU workflow.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/lib/cuda_env.sh"
rustinfer_discover_cuda_libraries
ARTIFACT_DIR="${ARTIFACT_DIR:-$(mktemp -d /tmp/rustinfer-v4.XXXXXX)}"
mkdir -p "$ARTIFACT_DIR"
cargo test --profile "${V4_TEST_PROFILE:-release}" --locked -p infer-backend-cuda \
    --test v4_swa --test v4_hca --test v4_csa --test v4_indexer --test v4_indexer_topk \
    -- --ignored --test-threads=1 2>&1 | tee "$ARTIFACT_DIR/v4-kernels.log"
