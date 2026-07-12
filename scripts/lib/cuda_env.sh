# Shared CUDA runtime-library discovery for RustInfer launch scripts.
# Source this file, then call rustinfer_discover_cuda_libraries.

rustinfer_prepend_library_path() {
    local directory="$1"
    [[ -d "$directory" ]] || return 0

    case ":${LD_LIBRARY_PATH:-}:" in
        *":${directory}:"*) ;;
        *) LD_LIBRARY_PATH="${directory}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
}

rustinfer_discover_cuda_libraries() {
    local candidate
    local component_dir
    local -a candidates=(
        "/usr/local/cuda/lib64"
        "/usr/local/cuda/lib"
    )

    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("${CUDA_HOME}/lib64" "${CUDA_HOME}/lib")
    fi
    if [[ -n "${CUDA_PATH:-}" ]]; then
        candidates+=("${CUDA_PATH}/lib64" "${CUDA_PATH}/lib")
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        # Prefer CUDA-specific directories. `${CONDA_PREFIX}/lib` may also
        # contain unrelated libraries (for example libcurl) that would shadow
        # the host tools used by the launch/e2e scripts.
        candidates+=("${CONDA_PREFIX}/targets/x86_64-linux/lib")
        for component_dir in "${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/*/lib; do
            [[ -d "$component_dir" ]] && candidates+=("$component_dir")
        done
    fi

    if command -v python3 >/dev/null 2>&1; then
        while IFS= read -r candidate; do
            [[ -n "$candidate" ]] && candidates+=("$candidate")
        done < <(python3 - <<'PY'
import importlib.util
from pathlib import Path

for module in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn"):
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        # Namespace-package parents may be absent from this interpreter even
        # when another Python installation owns the CUDA wheel.
        continue
    if spec is None:
        continue
    roots = spec.submodule_search_locations
    if roots is None and spec.origin:
        roots = [str(Path(spec.origin).parent)]
    for root in roots or ():
        print(Path(root) / "lib")
PY
        )
    fi

    for candidate in "${candidates[@]}"; do
        if compgen -G "${candidate}/libcublas.so*" >/dev/null \
            || compgen -G "${candidate}/libcudart.so*" >/dev/null \
            || compgen -G "${candidate}/libcudnn.so*" >/dev/null; then
            rustinfer_prepend_library_path "$candidate"
        fi
    done

    export LD_LIBRARY_PATH
}
