#!/bin/bash
#
# Start RustInfer Worker (foreground)
#
# Usage:
#   ./start_worker.sh [config_path]
#
# Environment variables (override positional args):
#   CONFIG    Path to the shared TOML config (default: rustinfer.toml)
#
# Startup order: run AFTER start_scheduler.sh, then run start_api.sh
#

set -e

CONFIG="${CONFIG:-${1:-rustinfer.toml}}"

# Ensure CUDA runtime libs (libcublas.so.12, etc.) are on the linker path.
# Portable across machines: probe common CUDA / conda / pip-wheel locations and
# prepend the first dir that actually contains libcublas.so.12.
if ! ldconfig -p 2>/dev/null | grep -q 'libcublas\.so\.12'; then
    _cuda_candidates=(
        "$CUDA_HOME/lib64" "$CUDA_HOME/lib"
        "$CONDA_PREFIX/lib"
        "/usr/local/cuda/lib64"
    )
    # pip nvidia wheels (torch-style), if a python is around
    if command -v python >/dev/null 2>&1; then
        _wheel="$(python -c 'import os,nvidia.cublas as c;print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)"
        [ -n "$_wheel" ] && _cuda_candidates+=("$_wheel")
    fi
    for _d in "${_cuda_candidates[@]}"; do
        if [ -n "$_d" ] && [ -e "$_d/libcublas.so.12" ]; then
            export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
            echo "  [cuda] libcublas found in $_d"
            break
        fi
    done
fi

echo "═════════════════════════════════════════════════════"
echo "  RustInfer Worker"
echo "═════════════════════════════════════════════════════"
echo "  Config: $CONFIG"
echo "═════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec cargo run --release -p infer-worker --bin rustinfer-worker -- --config "$CONFIG"
