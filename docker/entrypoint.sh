#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  rustinfer-container serve [options]
  rustinfer-container <command> [args...]

Options:
  --model PATH       Model directory inside the container.
  --model-name NAME  Name exposed by the OpenAI-compatible API.
  --device DEVICE    CUDA device, default: cuda:0.
  --host HOST        HTTP bind address, default: 0.0.0.0.
  --port PORT        HTTP port, default: 8000.
  --config PATH      Use a complete mounted TOML config instead of generating one.
  -h, --help         Show this help.

The same values can be supplied with RUSTINFER_MODEL, RUSTINFER_MODEL_NAME,
RUSTINFER_DEVICE, RUSTINFER_HOST, RUSTINFER_PORT, and RUSTINFER_CONFIG.
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "Missing value for ${option}" >&2
        exit 2
    fi
}

toml_escape() {
    local value="$1"
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        echo "TOML string values cannot contain newlines" >&2
        exit 2
    fi
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "$value"
}

validate_uint() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "${name} must be an unsigned integer, got: ${value}" >&2
        exit 2
    fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Preserve normal container debugging: `docker run ... bash` or an explicit
# RustInfer binary bypasses the supervisor.
if [[ -n "${1:-}" && "${1}" != "serve" && "${1}" != --* ]]; then
    exec "$@"
fi
if [[ "${1:-}" == "serve" ]]; then
    shift
fi

model="${RUSTINFER_MODEL:-/models/model}"
model_name="${RUSTINFER_MODEL_NAME:-}"
device="${RUSTINFER_DEVICE:-cuda:0}"
host="${RUSTINFER_HOST:-0.0.0.0}"
port="${RUSTINFER_PORT:-8000}"
config="${RUSTINFER_CONFIG:-}"

while (($# > 0)); do
    case "$1" in
        --model)
            require_value "$1" "${2:-}"
            model="$2"
            shift 2
            ;;
        --model-name)
            require_value "$1" "${2:-}"
            model_name="$2"
            shift 2
            ;;
        --device)
            require_value "$1" "${2:-}"
            device="$2"
            shift 2
            ;;
        --host)
            require_value "$1" "${2:-}"
            host="$2"
            shift 2
            ;;
        --port)
            require_value "$1" "${2:-}"
            port="$2"
            shift 2
            ;;
        --config)
            require_value "$1" "${2:-}"
            config="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

validate_uint "port" "$port"
if ((port == 0 || port > 65535)); then
    echo "port must be between 1 and 65535, got: ${port}" >&2
    exit 2
fi

cluster_id="${RUSTINFER_CLUSTER_ID:-rustinfer}"
if [[ ! "$cluster_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUSTINFER_CLUSTER_ID may contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
fi

if [[ -n "$config" ]]; then
    if [[ ! -f "$config" ]]; then
        echo "Config file not found: ${config}" >&2
        exit 2
    fi
else
    if [[ ! -d "$model" ]]; then
        echo "Model directory not found: ${model}" >&2
        echo "Mount a Hugging Face model directory at /models/model or pass --model." >&2
        exit 2
    fi
    for required_file in config.json tokenizer.json; do
        if [[ ! -f "${model}/${required_file}" ]]; then
            echo "Model directory is missing ${required_file}: ${model}" >&2
            exit 2
        fi
    done

    max_batch_tokens="${RUSTINFER_MAX_BATCH_TOKENS:-4096}"
    max_batch_seqs="${RUSTINFER_MAX_BATCH_SEQS:-32}"
    max_model_len="${RUSTINFER_MAX_MODEL_LEN:-4096}"
    chunked_prefill_size="${RUSTINFER_CHUNKED_PREFILL_SIZE:-256}"
    request_timeout_secs="${RUSTINFER_REQUEST_TIMEOUT_SECS:-600}"
    for numeric_setting in \
        "RUSTINFER_MAX_BATCH_TOKENS:${max_batch_tokens}" \
        "RUSTINFER_MAX_BATCH_SEQS:${max_batch_seqs}" \
        "RUSTINFER_MAX_MODEL_LEN:${max_model_len}" \
        "RUSTINFER_CHUNKED_PREFILL_SIZE:${chunked_prefill_size}" \
        "RUSTINFER_REQUEST_TIMEOUT_SECS:${request_timeout_secs}"; do
        validate_uint "${numeric_setting%%:*}" "${numeric_setting#*:}"
    done

    config="/tmp/rustinfer-container.toml"
    cat >"$config" <<EOF
model = "$(toml_escape "$model")"
model_name = "$(toml_escape "$model_name")"
cluster_id = "$(toml_escape "$cluster_id")"
device = "$(toml_escape "$device")"
host = "$(toml_escape "$host")"
port = ${port}
request_timeout_secs = ${request_timeout_secs}
max_batch_tokens = ${max_batch_tokens}
max_batch_seqs = ${max_batch_seqs}
max_model_len = ${max_model_len}
batch_wait_ms = 0
paged_block_size = 1
chunked_prefill_size = ${chunked_prefill_size}
enable_prefix_caching = false
mem_fraction_static = 0.85
num_blocks = 0
ignore_eos = false
mode = "llm"
worker_id = "worker-0"
log_level = "$(toml_escape "${RUST_LOG:-info}")"
capture_sizes = [1, 2, 4, 8, 16, 24, 32]
EOF
fi

for socket in \
    "/tmp/rustinfer-${cluster_id}-frontend.ipc" \
    "/tmp/rustinfer-${cluster_id}-worker-in.ipc" \
    "/tmp/rustinfer-${cluster_id}-worker-out.ipc" \
    "/tmp/rustinfer-${cluster_id}-worker-control.ipc"; do
    rm -f "$socket"
done

echo "RustInfer container starting"
echo "  build CUDA arch: ${RUSTINFER_CUDA_ARCH:-unknown}"
echo "  config:          ${config}"
echo "  API:             http://${host}:${port}"
printf '%s' "$port" >/tmp/rustinfer-port

pids=()

terminate_children() {
    trap - TERM INT
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    local attempt
    for attempt in {1..50}; do
        local running=false
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                running=true
                break
            fi
        done
        [[ "$running" == false ]] && break
        sleep 0.1
    done

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
}

on_signal() {
    echo "RustInfer container stopping"
    terminate_children
    exit 143
}

trap on_signal TERM INT

rustinfer-scheduler --config "$config" &
pids+=("$!")
rustinfer-worker --config "$config" &
pids+=("$!")
rustinfer-server --config "$config" &
pids+=("$!")

set +e
wait -n "${pids[@]}"
status=$?
set -e

echo "A RustInfer process exited with status ${status}; stopping the container" >&2
terminate_children
if ((status == 0)); then
    exit 1
fi
exit "$status"
