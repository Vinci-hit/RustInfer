# RustInfer

A high-performance, **architecture-first** LLM inference engine written in Rust.
OpenAI-compatible API, continuous batching, paged KV cache, and CUDA-graph decode
— built on a hexagonal, zero-cost multi-backend core that swaps CUDA for CPU at
compile time with no runtime penalty.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-H200-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Performance

RustInfer **outperforms vLLM** on an online QPS sweep — **Qwen3-4B, NVIDIA H200**,
`max_tokens=512`, `ignore_eos`, matched CUDA-graph decode capture sizes. Across the
sweep, RustInfer (red) holds lower TTFT / TPOT / ITL and lower end-to-end latency
than vLLM (blue) at equal or higher throughput:

![RustInfer vs vLLM — online QPS sweep, H200](bench/plots/ri_vs_vllm_qps_final.png)

> **RustInfer beats vLLM on the tail, not just the median.** Tail inter-token
> latency (**ITL p99**) stays below vLLM at *every* arrival rate — **6.6 → 9.0 ms**
> vs **7.2 → 10.9 ms** (qps 1 → 32) — alongside lower ITL / TPOT median and
> end-to-end latency at matched or higher throughput. Bench harness under `bench/`.

---

## Design philosophy

RustInfer is organized around a few principles, applied consistently top to bottom.

### Hexagonal core (ports & adapters)

`infer-core` owns nothing but **ports** — trait definitions for everything the
inference path needs from hardware:

```
infer-core/ports/
  backend.rs      math_ops.rs     fused_ops.rs
  sampler.rs      collective.rs   op_ports.rs
```

The backends are **adapters** that implement those ports: `infer-backend-cuda`
(`.cu` kernels + cuBLASLt + CUTLASS) and `infer-backend-cpu` (a pure-Rust reference
implementation, always linked, used as baseline and for tests). The core has zero
knowledge of CUDA; the entire GPU toolchain (nvcc / bindgen / cuDNN / CUTLASS) is
confined to the single `infer-backend-cuda` leaf crate.

### Heterogeneous backends at zero cost

The model layer is generic over an `LlmBackend` trait and **monomorphizes at
compile time** to whichever backend is selected — CUDA or CPU. There is no virtual
dispatch on the inference hot path: dispatch cost is paid by the compiler, not per
op. The same model code runs on GPU in production and on the CPU reference backend
in unit tests, byte-for-byte the same call sites.

### High cohesion, low coupling

Eight crates form an acyclic dependency graph with a GPU-free bottom. Each crate
has one job; cross-crate contact happens only through `infer-protocol` (wire types)
and `infer-core` (ports). Swapping a backend, a scheduler policy, or a transport
touches exactly one crate.

### DDD layering inside the worker

The worker — the most complex crate — is split into Domain / Application /
Infrastructure, so pure inference logic never mixes with I/O or orchestration:

```
infer-worker/src/
  domain/          model.rs, plan.rs, kv, forward_scratch, global_kv_alloc
                   → pure inference logic; no I/O, no transport
  application/     runtime, decode_engine, serve_loop, worker_scheduler,
                   sampler_stack, hosting  → orchestration & lifecycle
  infrastructure/  io, transport           → ZMQ / MsgPack adapters
  components/      attention, ffn, norm, embed, lm_head  → reusable NN blocks
  models/          llama3, qwen3, decoder, loader        → composition
```

### Model variation lives in the data, not in branches

A model's specialness (quantization, hybrid attention, tied embeddings) is an
**attribute of the operator/weight it lives on**, not a conditional threaded through
higher layers. The weight loader is generic and name-driven — it reads exactly the
tensor names and shapes the checkpoint declares; only the model module knows how to
assemble them.

---

## Architecture

Three cooperating processes share a single TOML config and communicate over
ZMQ (IPC) with MessagePack framing:

```
  infer-server              infer-scheduler                infer-worker
  ┌──────────────┐          ┌──────────────────┐          ┌────────────────────┐
  │ Axum /v1/... │          │ RadixTree prefix   │          │ Runtime<T,D,Model>  │
  │ chat template│  ZMQ     │   cache            │  ZMQ     │  ├ persistent ABC   │
  │ tokenizer    │ ───────► │ continuous batching│ ───────► │  ├ CUDA-graph capture│
  │ SSE stream   │ ◄─────── │ chunked prefill    │ ◄─────── │  └ KV / scratch pool │
  └──────────────┘          └──────────────────┘          │        │            │
         │                          │                       │        ▼            │
         └──────────┬───────────────┘                       │  DecoderModel       │
                    ▼                                        │  (Decoder<T,D>)     │
            infer-protocol                                   │        │            │
  (config / server↔sched / sched↔worker msgs)                │        ▼            │
                                                             │  Components         │
                                                             │ (Attention/FFN/...) │
                                                             └────────┬───────────┘
                                                                      │ calls ops
                                                                      ▼
                                                   infer-core ── LlmBackend (trait port)
                                                                      ▲        ▲
                                                        impl          │        │  impl
                                                  ┌───────────────────┘        └──────────┐
                                          infer-backend-cuda            infer-backend-cpu
                                          (.cu kernels + cuBLASLt)      (reference / tests)
```

### Workspace

| Crate                 | Role |
|-----------------------|------|
| `infer-core`          | Foundation: dtypes, quant scheme, value types, and the `LlmBackend` **ports**. GPU-free — the bottom of the DAG. |
| `infer-protocol`      | Wire types: config parsing + server↔scheduler↔worker messages. |
| `infer-server`        | Axum HTTP front end, OpenAI `/v1` API, chat template, SSE streaming. |
| `infer-scheduler`     | RadixTree prefix cache, continuous batching, chunked prefill, batch planning. |
| `infer-worker`        | GPU inference runtime (DDD: domain / application / infrastructure), models, components. |
| `infer-backend-cuda`  | CUDA adapter: `.cu` kernels + cuBLASLt; statically links the kernel set + CUTLASS. |
| `infer-backend-cpu`   | CPU adapter: pure-Rust reference backend; always linked as baseline / fallback. |
| `infer-frontend`      | Optional front end (outside the core inference path). |

---

## Features

- **OpenAI-compatible API** — `/v1/chat/completions` and `/v1/completions`, with
  SSE streaming, chat templates, and HF tokenizers.
- **Continuous batching** with chunked prefill and RadixTree prefix caching.
- **Paged KV cache** with profile-driven sizing and KV recycling.
- **CUDA-graph decode** — captured graphs over a fixed set of batch sizes, with a
  persistent ABC buffer that eliminates per-step allocation in the hot loop.
- **Quantization** — dense BF16 and AWQ int4 (W4A16) MLP.
- **Models** — Llama-3.2, Qwen3, Qwen3 (AWQ), and Qwen3.5-4B text/image
  inference at TP=1; see [Status](#status).

---

## Quick start

### Run the prebuilt Docker image (H100 / H200)

The public image contains the compiled CUDA kernels, CUDA 12 runtime, cuBLAS,
and cuDNN. The host only needs:

- an NVIDIA H100 or H200 with a compatible driver;
- Docker and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
- a local Hugging Face model directory containing `config.json`,
  `tokenizer.json`, and the model weights.

Start the full scheduler + worker + OpenAI-compatible server stack with one
command. Replace the host model path and exposed model name as needed:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /absolute/path/to/Qwen3:/models/model:ro \
  ghcr.io/vinci-hit/rustinfer:1.0.1 \
  serve --model-name Qwen3
```

Wait for the model to load, then check readiness:

```bash
curl --fail http://127.0.0.1:8000/ready
```

Send a chat completion:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":64}'
```

No Rust toolchain, CUDA Toolkit, `nvcc`, libclang, cuDNN headers, or local
operator compilation is required. The `1.0.1` image is `linux/amd64` and
compiled for CUDA architecture `sm_90`.

Useful container settings include `RUSTINFER_MAX_BATCH_TOKENS`,
`RUSTINFER_MAX_BATCH_SEQS`, `RUSTINFER_MAX_MODEL_LEN`,
`RUSTINFER_CHUNKED_PREFILL_SIZE`, and `RUST_LOG`. A complete custom config can
instead be mounted and selected with `--config`.

### Build the Docker image locally

To build from the current source instead of using the published image:

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_ARCH=sm_90 \
  -t rustinfer:local .
```

Images are architecture-specific. Use `CUDA_ARCH=sm_80`, `sm_86`, or `sm_89`
when building for another supported NVIDIA GPU architecture.

### Build from source

#### Prerequisites

- Rustup (the repository pins Rust 1.91.1), a CUDA-capable GPU, and the CUDA
  toolkit.
- The cuDNN frontend headers on the include path:

```bash
export CUDNN_FRONTEND_INCLUDE_DIR=/path/to/site-packages/include
```

#### Build

```bash
cargo build --release
```

Optional [Triton RMSNorm kernels](docs/TRITON.md) are enabled with
`cargo build --release -p infer-worker --features triton`; see the guide for the
pinned Python build dependency and supported layouts.

Optional [TileLang RMSNorm kernels](docs/TILELANG.md) are enabled with
`cargo build --release -p infer-worker --features tilelang`.

#### Run (one-shot e2e smoke test)

Launches scheduler + worker + server for a config, sends one chat completion,
prints the reply, and tears everything down:

```bash
scripts/e2e_smoke.sh rustinfer.toml 8000 "Say hello in one short sentence."
```

The default launch config is `rustinfer.toml`. Set its `model` field to the
local Hugging Face model directory before launching; it binds to
`127.0.0.1:8000` and uses `cuda:0` by default.

#### Run (manual, three processes)

Each binary takes the same `--config`:

```bash
./target/release/rustinfer-scheduler --config rustinfer.toml &
./target/release/rustinfer-worker    --config rustinfer.toml &
./target/release/rustinfer-server     --config rustinfer.toml &
```

Then hit the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model",
       "messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Use the model ID returned by `/v1/models` in the request's `model` field.

#### Python benchmark tools

The Python project keeps benchmark dependencies optional so building RustInfer
does not install another inference engine. Install only the group a script uses:

```bash
uv sync --extra bench
uv sync --extra vllm-reference
uv sync --extra sglang-reference
```

The vLLM and SGLang groups are reference-benchmark environments; they are not
runtime dependencies of RustInfer.

---

## Configuration

Config is a single TOML shared by all three processes. Key fields:

| Field                   | Meaning |
|-------------------------|---------|
| `model`                 | Path to the HF model directory (config + safetensors + tokenizer). |
| `model_name`            | Name reported by the `/v1` API. |
| `device`                | Rank-0 CUDA device, e.g. `cuda:0`; TP ranks use consecutive devices. |
| `tensor_parallel_size`  | Number of GPUs in the single-process TP group (`1` = disabled). |
| `port`                  | HTTP port for the server. |
| `tp_operation_timeout_secs` | Fail-stop deadline for one mirrored TP inference operation. |
| `tp_startup_timeout_secs` | Longer fail-stop deadline for NCCL and follower startup. |
| `max_batch_tokens`      | Token budget per forward batch. |
| `max_batch_seqs`        | Max concurrent sequences in a batch. |
| `max_model_len`         | Max context length. |
| `chunked_prefill_size`  | Chunked-prefill chunk size (`0` = disabled). |
| `enable_prefix_caching` | Toggle the RadixTree prefix cache. |
| `mem_fraction_static`   | Fraction of GPU memory reserved for weights + static buffers. |
| `num_blocks`            | KV-cache blocks (`0` = auto-size from a memory profile). |
| `capture_sizes`         | Batch sizes to capture CUDA graphs for, e.g. `[1,2,4,8,16,24,32]`. |
| `mtp_num_draft_tokens`  | Opt-in Qwen3.5 text MTP; `0` disables it. Requires greedy sampling, TP1, `max_batch_seqs=1`, and prefix caching disabled. Verification runs eager. |
| `ignore_eos`            | Ignore EOS (useful for fixed-length benchmarking). |

### Tensor parallelism

Tensor parallelism is disabled by default. To shard a dense BF16 model over two
GPUs, set the shared configuration to:

```toml
device = "cuda:0"
tensor_parallel_size = 2
```

One worker process owns the complete TP group. Rank 0 uses `device`; the other
ranks use consecutive CUDA device IDs, so this example uses `cuda:0` and
`cuda:1`. The CUDA backend requires NCCL 2.24.3 or newer; the Docker image
already includes the matching development and runtime packages.

The implementation supports single-node dense BF16 and block-FP8 Llama/Qwen
decoders. Vocabulary size, query/KV head counts, MLP intermediate size, and FP8
weight/scale block boundaries must be evenly divisible by
`tensor_parallel_size`.

CUDA Graph capture is enabled for TP decode and single-sequence prefill. Every
rank captures its own device graph and replays the same NCCL collective sequence
in lockstep. Mixed prefill+decode batches currently stay eager. TP with AWQ
weights, speculative decoding, pipeline parallelism, data parallelism, and
expert parallelism are not implemented yet; unsupported combinations fail
during startup instead of silently falling back to replicated execution.

---

## Status

RustInfer serves Llama-3.2, Qwen3, and Qwen3-AWQ end to end today.

Qwen3.5-4B runs hybrid Gated DeltaNet/full attention, text and PNG/JPEG data-URL
inputs, chunked prefill, and CUDA Graph decode at TP=1. The reproducible service
regression covers mixed text/image concurrency and streaming. Hardware/model
combinations still require GPU validation; a successful CUDA build alone does
not establish inference correctness.

DeepSeek V4 has an offline [tiny-model validation path](docs/DEEPSEEK_V4_TINY.md)
for a single 16GB GPU. It checks unquantized random weights against Transformers
using CUDA GEMMs and host reference operators; full-checkpoint serving is pending.
Independent [GPU SWA operators](docs/DEEPSEEK_V4_SWA.md) provide
BF16 shared-KV attention with a 128-token ring cache, 512-dimensional heads,
attention sinks and CUDA Graph replay. Decode uses one fused kernel; Tensor Core
prefill supports full/chunked inputs followed by a stream-ordered cache commit.
Both paths allocate no temporary GPU storage.
The [GPU HCA operators](docs/DEEPSEEK_V4_HCA.md) add incremental 128-token
compression with 6 KiB of FP32 state, plus joint local/compressed attention for
decode and Tensor Core prefill. They support chunked inputs and CUDA Graphs;
model execution integration and checkpoint quantization are pending.
The [GPU CSA compressor](docs/DEEPSEEK_V4_CSA.md) adds overlapping 4-token
compression, fused pooling/RMSNorm/RoPE, and 18 KiB of incremental FP32 state.
It supports full/chunked prefill and graph decode.
The [Lightning Indexer scoring kernel](docs/DEEPSEEK_V4_INDEXER.md) fuses BF16
Tensor Core dot products, ReLU, FP32 weighted head reduction and causal masking.
Its [deterministic top-k operator](docs/DEEPSEEK_V4_TOPK.md) selects up to 512
compressed-row IDs using block radix selection and parallel merges, with
caller-owned graph-safe scratch. Index-key preparation and sparse joint
attention remain pending.

Workers execute a bounded prefill/decode self-check before advertising ready.
`/health` reports HTTP process liveness; `/ready` requires a loaded Worker group
and a fresh scheduler engine heartbeat, and becomes unavailable during failure
or shutdown. `/metrics` exports Prometheus text; `/metrics/system` retains the
browser console's JSON uptime summary. See [metric semantics](docs/METRICS.md).

Run the local GPU regression, or provision the automatic master/nightly GPU
workflow using [the validation guide](docs/VALIDATION.md):

```bash
MODEL_PATH=/absolute/path/to/Qwen3.5-4B \
ARTIFACT_DIR=/tmp/rustinfer-gpu-run-001 \
bash scripts/gpu_regression.sh
```

---

## License

Licensed under the [Apache License 2.0](LICENSE). Vendored components retain
their own terms.
