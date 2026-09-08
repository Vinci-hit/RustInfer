# Request limits and validation

The chat and text completion endpoints acquire the configured HTTP admission
permit before reading the request body. Overload returns HTTP 429. The permit
remains held during inference, streaming, and non-cancellable background
preprocessing; extraction errors release it.

Chat bodies allow up to 56 MiB to accommodate base64 images. Chat text (the sum
of role strings and text parts) and completion text prompts are limited to
1 MiB of UTF-8 each, returning HTTP 400 when exceeded. Stop strings have a
separate aggregate 1 MiB limit. Image URLs do not count toward the text limit;
the existing image count, encoded size, pixel, and visual-token limits still
apply. Text completions retain Axum's default 2 MiB body limit. Prompt and stop
string tokenization run on blocking workers while retaining admission permits.

## Startup and readiness

The Worker runs a bounded prefill/decode self-check with its final KV pool and
graph configuration before advertising Ready. A failure aborts startup with
the operation error. `/health` reports HTTP process liveness; `/ready` requires
both Worker/model startup and a fresh scheduler engine heartbeat. Loading,
draining, failed, stale, and protocol-incompatible states return HTTP 503.
Readiness is sampled over periodic Pongs, so failure detection is bounded by
the worker watchdog and heartbeat freshness windows, not instantaneous.

This changes the server/scheduler frontend protocol to version 3. Upgrade both
processes together; a mixed-version deployment remains unready. It does not
change the worker control-plane version. Readiness withdrawal on shutdown is
implemented; an ordered, deadline-bounded drain of all three processes remains
separate work.

## CI

The normal CI workflow checks CPU crates and builds the Docker `builder` stage
on every push and pull request. This compiles and links the production CUDA
worker and kernels for `sm_90` without requiring a GPU. It does not publish an
image or establish numerical correctness.

The **GPU validation** workflow runs on pushes to `master` and daily at 19:00
UTC once `GPU_CI_ENABLED=true` is configured. It also supports manual dispatch.
Provision a trusted self-hosted Linux x64 runner labelled `gpu`, with Docker,
NVIDIA Container Toolkit, and an NVIDIA GPU; untrusted pull requests never run
on this persistent runner. Set these repository variables:

| Variable | Meaning |
| --- | --- |
| `GPU_CI_ENABLED` | Set to `true` after provisioning to enable automatic runs. |
| `GPU_CUDA_ARCH` | Architecture matching the runner, e.g. `sm_89` or `sm_90`; defaults to `sm_90`. |
| `QWEN35_MODEL_PATH` | Absolute path to local Qwen3.5-4B weights on the runner. |

An unconfigured repository skips automatic GPU jobs. Manual runs still require
the runner and model directory; choose the matching architecture in the form.
The workflow builds the Docker development stage and runs:

- BF16 Gated DeltaNet CPU/CUDA comparison across prefill and decode.
- HD256 paged attention across split query tiles, cached prefixes, and padded
  graph tiles, including the 96 KiB shared-memory path used on Ada GPUs.
- The full Qwen3.5 service regression described below.

Build logs, process logs, launch configuration, metrics, environment metadata,
and request/response results are uploaded even when a test fails. Artifacts are
retained for 14 days. These fixtures use synthetic inputs, never live traffic.

To also run the ignored image preprocessing reference tests, enable
`vision_reference` and set repository variables `QWEN35_MODEL_PATH` and
`QWEN35_VISION_REFERENCE` to directories on the runner. The first contains the
model and tokenizer; the second contains prepared reference fixtures required
by `crates/infer-server/src/chat/multimodal.rs`. The reference directory must be
writable because the tests export the Rust patch bytes. These optional tests
compare preprocessing, not full model generation accuracy.

## Local GPU regression

The same suite runs without Docker in an existing CUDA development environment:

```bash
MODEL_PATH=/absolute/path/to/Qwen3.5-4B \
ARTIFACT_DIR=/tmp/rustinfer-gpu-run-001 \
bash scripts/gpu_regression.sh
```

To run only service checks after compiling the binaries:

```bash
python3 scripts/e2e_qwen35_smoke.py \
  --model /absolute/path/to/Qwen3.5-4B \
  --output /tmp/rustinfer-service-run-001
```

Use a fresh artifact directory for each run. The Python script requires no
third-party packages. It selects an available local HTTP port and unique IPC
namespace, writes its launch configuration, and always stops its own processes.
It checks loading readiness, successful startup, text/image generation, SSE,
mixed concurrency, oversized/invalid inputs, cancellation, Prometheus output,
CUDA Graph replay, and readiness withdrawal after Worker loss. A 512-token
output budget with a 256-token batch budget also covers long-generation
admission. It is a semantic/service regression, not an HF precision comparison
or a hardware-independent performance benchmark.

`environment.json` records the revision, working-tree status, binary hashes,
GPU/driver, Rust version, and SHA-256
of model configuration, tokenizer, processor configuration, and weight index.
These metadata hashes do not checksum the weight shards themselves. Keep the
checkpoint directory immutable between comparisons. `results.json` records
each completed request and its elapsed time; process logs retain failure detail.

See [METRICS.md](METRICS.md) for exporter names and measurement semantics.
