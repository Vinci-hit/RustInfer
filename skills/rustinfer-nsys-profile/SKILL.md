---
name: rustinfer-nsys-profile
description: This skill should be used when profiling RustInfer performance with Nsight Systems, especially worker-only CUDA profiling, cudaProfilerApi capture ranges, paged KV / prefix-cache benchmarks, or diagnosing whether bottlenecks are in Worker GPU kernels rather than Scheduler/HTTP control plane.
---

# RustInfer Nsight Systems Profiling

## Purpose

Profile RustInfer performance by isolating the `rustinfer-worker` process, because CUDA kernels execute in Worker while HTTP, Scheduler, and ZMQ mostly represent control-plane overhead. Use this workflow when investigating throughput, latency, CUDA kernel time, paged KV bottlenecks, CUDA Graph behavior, or nsys trace quality.

## Core Rule

Prefer worker-only profiling. Do not profile the whole launcher unless the task is specifically about control-plane overhead. Start Scheduler and Server normally, but launch Worker under `nsys profile`.

## Recommended cudaProfilerApi Workflow

Use Worker's built-in profiler range:

```bash
--profile-cuda-steps N
```

This starts CUDA profiling on the first submitted Worker step via `cudaProfilerStart()` and stops after `N` completed Worker steps via `cudaProfilerStop()`. Pair it with:

```bash
--capture-range=cudaProfilerApi
--capture-range-end=stop-shutdown
```

This avoids guessing `--delay` and captures only request-stage CUDA work.

### 1. Start Scheduler

```bash
cd /root/RustInfer
mkdir -p result
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b

PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-scheduler \
  --frontend-endpoint ipc:///tmp/rustinfer-nsys-frontend.ipc \
  --worker-push-endpoint ipc:///tmp/rustinfer-nsys-worker-in.ipc \
  --worker-pull-endpoint ipc:///tmp/rustinfer-nsys-worker-out.ipc \
  --worker-control-endpoint ipc:///tmp/rustinfer-nsys-worker-control.ipc \
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

### 2. Start Worker Under nsys

```bash
PROFILE_STEPS=200

PATH=/root/RustInfer/target/release:$PATH \
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --cuda-trace-all-apis=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --sample=none \
  --cpuctxsw=none \
  --kill=none \
  --force-overwrite=true \
  --output=/root/RustInfer/result/nsys_paged_worker \
  ./target/release/rustinfer-worker \
    --device cuda:0 \
    --worker-pull-endpoint ipc:///tmp/rustinfer-nsys-worker-in.ipc \
    --worker-push-endpoint ipc:///tmp/rustinfer-nsys-worker-out.ipc \
    --worker-control-endpoint ipc:///tmp/rustinfer-nsys-worker-control.ipc \
    --max-batch-tokens 512 \
    --max-batch-seqs 4 \
    --profile-cuda-steps ${PROFILE_STEPS} \
    --log-level warn
```

### 3. Start Server and Drive Workload

```bash
PATH=/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-server \
  --port 8014 \
  --engine-endpoint ipc:///tmp/rustinfer-nsys-frontend.ipc \
  --tokenizer ${MODEL} \
  --model-name llama3.2-1b \
  --log-level warn

python bench/bench_real_arrival.py \
  --url http://127.0.0.1:8014 \
  --model llama3.2-1b \
  --label RustInfer-PagedPrefix-Llama3.2-1B-nsys \
  --warmup-requests 2 \
  --num-requests 8 \
  --concurrency 2 \
  --arrival-rate 2 \
  --max-tokens 32 \
  --seed 20260521 \
  --verbose
```

### 4. Generate nsys Stats

```bash
rm -f /root/RustInfer/result/nsys_paged_worker.sqlite
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum \
  /root/RustInfer/result/nsys_paged_worker.nsys-rep
```

## Delay/Duration Fallback

Use delay/duration only when `--profile-cuda-steps` is unavailable or disabled. Set `--delay` long enough to skip model loading and Paged KV allocation. If delay is wrong, nsys may only capture initialization kernels like `sin_cos_calc_bf16` or large H2D copies.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --cuda-trace-all-apis=true \
  --delay=60 \
  --duration=120 \
  --sample=none \
  --cpuctxsw=none \
  --kill=none \
  --force-overwrite=true \
  --output=/root/RustInfer/result/nsys_paged_worker_delay \
  ./target/release/rustinfer-worker ...
```

## Operator-Level Sanity Check

Before trusting serving traces, verify nsys can capture CUDA kernels with a deterministic operator test:

```bash
cd /root/RustInfer
cargo test -p infer-worker --features cuda,models --no-run
BIN=$(ls -t target/debug/deps/infer_worker-* | grep -v '\.d$' | head -n1)

nsys profile \
  --trace=cuda,cublas,nvtx,osrt \
  --cuda-trace-all-apis=true \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=/root/RustInfer/result/nsys_op_decode \
  ${BIN} test_flash_attn_decode_batch --nocapture

nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  /root/RustInfer/result/nsys_op_decode.nsys-rep
```

## Interpreting Results

Prioritize `cuda_gpu_kern_sum` for actual Worker GPU bottlenecks. In current paged correctness-first implementation, expected bottlenecks are:

- `paged_decode_naive_kernel<bf16>` dominating decode GPU time.
- `paged_ragged_naive_kernel<bf16>` dominating prefill/mixed GPU time.
- `scatter_kv_paged_kernel`, RMSNorm, RoPE, and SwiGLU should be much smaller.

Treat large `cudaMemcpy` API time carefully: it can include synchronization wait behind slow kernels. Confirm with GPU kernel summary before optimizing copies.

## Cleanup

After profile runs, clean service processes and verify GPU state:

```bash
pkill -TERM -f 'rustinfer-(scheduler|worker|server).*nsys|rustinfer-nsys' || true
sleep 2
pkill -KILL -f 'rustinfer-(scheduler|worker|server).*nsys|rustinfer-nsys' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## Keep README in Sync

The same profiling workflow is documented in `/root/RustInfer/README.md` under `Nsight Systems Profile 方法`. Update both this skill and README when profile flags, ports, endpoints, or benchmark commands change.
