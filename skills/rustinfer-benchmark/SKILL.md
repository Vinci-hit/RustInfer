---
name: rustinfer-benchmark
description: This skill should be used when running RustInfer benchmarks, comparing RustInfer with vLLM, testing paged KV / prefix cache behavior, measuring real-arrival online workloads, or validating GPU/HTTP throughput and latency after inference changes.
---

# RustInfer Benchmark Skill

## Purpose

Run RustInfer benchmarks in a repeatable way, separate runner-level GPU performance from HTTP/control-plane performance, and compare against vLLM using equivalent model, prompt, token, concurrency, and arrival settings.

## Core Principles

- Use absolute paths from `/root/RustInfer`.
- Clean stale GPU/process state before each benchmark.
- Do not use streaming benchmarks for throughput unless explicitly requested.
- Use `temperature=0.0` for deterministic comparisons.
- For decode throughput comparisons, force fixed output length where supported using `min_tokens=max_tokens` and `ignore_eos=true`.
- Distinguish three layers:
  1. runner-only decode benchmark: pure Worker/ModelRunner GPU path;
  2. online HTTP benchmark: full RustInfer control plane;
  3. vLLM OpenAI-compatible benchmark: external baseline.
- For paged KV tests, run correctness/smoke first, then benchmark. Correctness-first paged kernels are expected to be slow.

## Pre-Benchmark Cleanup

Run before any benchmark:

```bash
ps -eo pid,cmd | grep -E 'rustinfer|vllm.entrypoints.openai|api_server|bench_real_arrival' | grep -v grep || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

If stale RustInfer/vLLM processes exist:

```bash
pkill -TERM -f 'rustinfer-(scheduler|worker|server)|rustinfer-launch|vllm.entrypoints.openai.api_server|api_server' || true
sleep 3
pkill -KILL -f 'rustinfer-(scheduler|worker|server)|rustinfer-launch|vllm.entrypoints.openai.api_server|api_server' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

If `nvidia-smi` shows `[Not Found]` PIDs that do not exist in `ps`, note them as orphan/NVML artifacts; do not invent cleanup commands outside visible process state.

## Build Release Binaries

```bash
cd /root/RustInfer
cargo build --release \
  -p infer-worker \
  -p infer-scheduler \
  -p infer-server \
  --features infer-worker/cuda,infer-worker/models
```

## Runner-Only Decode Matrix

Use this when comparing raw GPU decode path to `scripts/bench_vllm_decode.py`.

### RustInfer

For Qwen3:

```bash
cd /root/RustInfer
QWEN3_MODEL_PATH=/apdcephfs_qy2/share_303432435/vinciiliu/models/checkpoint-800-1 \
cargo test -p infer-worker --release --features cuda,models \
  tests_perf::perf_qwen3_decode_matrix -- --ignored --nocapture
```

For Llama3:

```bash
cd /root/RustInfer
LLAMA3_MODEL_PATH=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
cargo test -p infer-worker --release --features cuda,models \
  tests_perf::perf_llama3_decode_matrix -- --ignored --nocapture
```

### vLLM Decode Matrix

```bash
cd /root/RustInfer
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python scripts/bench_vllm_decode.py \
  --model /apdcephfs_qy2/share_303432435/vinciiliu/models/checkpoint-800-1 \
  --label checkpoint-800-1 \
  --decode-steps 256 \
  --batches 1,2,4,8 \
  --max-model-len 1024 \
  --gpu-mem-util 0.5 \
  --dtype bfloat16
```

Use `VLLM_WORKER_MULTIPROC_METHOD=spawn` to avoid CUDA reinitialization errors in forked subprocesses.

## RustInfer Online HTTP Benchmark

Use `bench/bench_real_arrival.py` for realistic online workloads:

- mixed prompt lengths;
- Poisson inter-arrival times;
- warmup excluded from final stats;
- OpenAI-compatible `/v1/chat/completions` endpoint.

### Slot KV Example

Start services manually:

```bash
cd /root/RustInfer
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b

PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-scheduler \
  --frontend-endpoint ipc:///tmp/rustinfer-bench-frontend.ipc \
  --worker-push-endpoint ipc:///tmp/rustinfer-bench-worker-in.ipc \
  --worker-pull-endpoint ipc:///tmp/rustinfer-bench-worker-out.ipc \
  --worker-control-endpoint ipc:///tmp/rustinfer-bench-worker-control.ipc \
  --model ${MODEL} \
  --model-type llama3 \
  --device cuda:0 \
  --max-batch-tokens 512 \
  --max-batch-seqs 4 \
  --max-model-len 1024 \
  --kv-cache-mode slot \
  --log-level warn
```

Worker:

```bash
PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-worker \
  --device cuda:0 \
  --worker-pull-endpoint ipc:///tmp/rustinfer-bench-worker-in.ipc \
  --worker-push-endpoint ipc:///tmp/rustinfer-bench-worker-out.ipc \
  --worker-control-endpoint ipc:///tmp/rustinfer-bench-worker-control.ipc \
  --max-batch-tokens 512 \
  --max-batch-seqs 4 \
  --log-level warn
```

Server:

```bash
PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-server \
  --port 8013 \
  --engine-endpoint ipc:///tmp/rustinfer-bench-frontend.ipc \
  --tokenizer ${MODEL} \
  --model-name llama3.2-1b \
  --log-level warn
```

Wait for health:

```bash
curl -fsS http://127.0.0.1:8013/health
```

Run benchmark:

```bash
cd /root/RustInfer
python bench/bench_real_arrival.py \
  --url http://127.0.0.1:8013 \
  --model llama3.2-1b \
  --label RustInfer-Llama3.2-1B \
  --warmup-requests 3 \
  --num-requests 12 \
  --concurrency 4 \
  --arrival-rate 2 \
  --max-tokens 32 \
  --seed 20260521
```

### Paged KV + Prefix Cache Example

Use this when validating paged KV functionality. Expect low performance until paged attention kernels are optimized.

```bash
cd /root/RustInfer
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b

PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-scheduler \
  --frontend-endpoint ipc:///tmp/rustinfer-paged-bench-frontend.ipc \
  --worker-push-endpoint ipc:///tmp/rustinfer-paged-bench-worker-in.ipc \
  --worker-pull-endpoint ipc:///tmp/rustinfer-paged-bench-worker-out.ipc \
  --worker-control-endpoint ipc:///tmp/rustinfer-paged-bench-worker-control.ipc \
  --model ${MODEL} \
  --model-type llama3 \
  --device cuda:0 \
  --max-batch-tokens 512 \
  --max-batch-seqs 4 \
  --max-model-len 1024 \
  --kv-cache-mode paged:16 \
  --mem-fraction-static 0.05 \
  --enable-prefix-caching \
  --log-level warn
```

Use matching Worker/Server endpoints and run `bench_real_arrival.py` as above.

## vLLM OpenAI-Compatible Benchmark

Start vLLM:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8001 \
  --model /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --served-model-name llama3.2-1b \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --disable-log-stats
```

Run identical workload:

```bash
cd /root/RustInfer
python bench/bench_real_arrival.py \
  --url http://127.0.0.1:8001 \
  --model llama3.2-1b \
  --label vLLM-Llama3.2-1B \
  --warmup-requests 3 \
  --num-requests 12 \
  --concurrency 4 \
  --arrival-rate 2 \
  --max-tokens 32 \
  --seed 20260521
```

For prefix-cache fairness tests, either disable vLLM prefix caching if supported by the installed version, or use prompts with unique randomized leading prefixes. Otherwise vLLM may benefit from prefix reuse while RustInfer may be in a different cache mode.

## Long Output Benchmark

For decode-heavy scenarios use at least `512` output tokens:

```bash
python bench/bench_real_arrival.py \
  --url http://127.0.0.1:8013 \
  --model llama3.2-1b \
  --label RustInfer-Llama3.2-1B-512 \
  --warmup-requests 10 \
  --num-requests 32 \
  --concurrency 32 \
  --arrival-rate 4 \
  --max-tokens 512 \
  --seed 20260520
```

The benchmark script sends `temperature=0.0`, `min_tokens=max_tokens`, and `ignore_eos=true`; RustInfer ignores unsupported fields but vLLM uses them to force full-length output.

## Interpreting Results

Record these metrics:

- `OK/failed`
- output tokens
- wall time
- system throughput `tok/s`
- mean latency
- p50/p90/p99 latency
- prompt length range

Interpret carefully:

- Runner decode matrix measures GPU model path only.
- HTTP online benchmark includes scheduler, server, tokenizer, ZMQ, and response path.
- Paged correctness kernels are intentionally slow; use nsys before treating paged throughput as representative.
- Prefix cache can improve prefill-heavy workloads but can make comparisons unfair if one side has it enabled and the other does not.
- Short `max_tokens` workloads emphasize control-plane and prefill overhead; long outputs emphasize decode path.

## Common Failure Modes

- vLLM CUDA fork error: set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- RustInfer request timeout: check worker logs for fatal `StepOutput.error`, panic, or blocked `NeedBlocks`/`GrantBlocks` state.
- Stale GPU memory: check `nvidia-smi --query-compute-apps`; clean visible processes before rerunning.
- Paged mixed batch slow: expected while paged decode/ragged kernels are correctness-first.

## Cleanup

```bash
pkill -TERM -f 'rustinfer-(scheduler|worker|server)|rustinfer-launch|vllm.entrypoints.openai.api_server|api_server' || true
sleep 3
pkill -KILL -f 'rustinfer-(scheduler|worker|server)|rustinfer-launch|vllm.entrypoints.openai.api_server|api_server' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
