---
name: rustinfer-nsys-profile
description: This skill should be used when profiling RustInfer performance with Nsight Systems, especially worker-only CUDA profiling, cudaProfilerApi capture ranges, paged KV / prefix-cache benchmarks, CUDA Graph decode behavior, or diagnosing whether bottlenecks are in Worker GPU kernels rather than Scheduler/HTTP control plane.
---

# RustInfer Nsight Systems Profiling

## Purpose

Profile RustInfer performance by isolating the `rustinfer-worker` process, because CUDA kernels execute in Worker while HTTP, Scheduler, and ZMQ mostly represent control-plane overhead. Use this workflow when investigating throughput, latency, CUDA kernel time, paged KV bottlenecks, CUDA Graph behavior, or nsys trace quality.

## Core Rule

Prefer worker-only profiling. Only the Worker runs CUDA kernels, so wrap only the Worker in `nsys profile`. Scheduler and HTTP server run unprofiled.

## Architecture (IMPORTANT — read before profiling)

`rustinfer-server` is an **all-in-one launcher**. It does NOT take `--engine-endpoint`. On startup it spawns its own `rustinfer-scheduler` and `rustinfer-worker` child processes by **PATH name**, wiring them together with auto-generated pid-based IPC endpoints (`ipc:///tmp/rustinfer-<pid>-frontend.ipc`, etc.). There is no flag to attach the server to an externally-launched scheduler/worker.

Consequence: you cannot hand-launch a worker under nsys and then point the server at it. Instead, profile the worker by intercepting how the launcher spawns it.

### Binaries (from Cargo `[[bin]]`)
- `rustinfer-server` — `crates/infer-server/src/main.rs` (all-in-one launcher)
- `rustinfer-scheduler` — `crates/infer-scheduler/src/main.rs`
- `rustinfer-worker` — `crates/infer-worker/src/bin/worker_main.rs`

### Key CLI facts (current binaries — older skill text was stale)
- Scheduler has **no** `--kv-cache-mode`. It takes `--paged-block-size <N>` (paged is the only mode). It does take `--mem-fraction-static` and `--enable-prefix-caching`.
- Worker endpoint flags: canonical `--data-recv-endpoint` / `--data-send-endpoint` / `--control-endpoint`, with aliases `--worker-pull-endpoint` / `--worker-push-endpoint` / `--worker-control-endpoint` (the launcher uses the alias names). Worker uses `--max-seq-len` (not `--max-model-len`).
- Worker `--profile-cuda-steps N`: calls `cudaProfilerStart()` before the first decode step and `cudaProfilerStop()` after N decode steps, **then exits**. The launcher does NOT pass this flag — the nsys shim below appends it.
- Server forwards `--kv-cache-mode paged:<N>` → scheduler `--paged-block-size <N>` for backwards compat.

## Recommended Workflow: nsys worker shim + all-in-one launcher

Because the launcher spawns the worker by PATH name, put a shim named `rustinfer-worker` first in `PATH`. The shim `exec`s the real worker under `nsys profile` and appends `--profile-cuda-steps`. Scheduler and server resolve to the real binaries (second PATH entry).

### 1. Create the nsys worker shim (one-time)

`/root/RustInfer/result/nsys-shim/rustinfer-worker`:

```bash
#!/usr/bin/env bash
set -euo pipefail
REAL_WORKER="${RUSTINFER_REAL_WORKER:-/root/RustInfer/target/release/rustinfer-worker}"
NSYS_OUT="${RUSTINFER_NSYS_OUT:-/root/RustInfer/result/nsys_paged_worker}"
PROFILE_STEPS="${RUSTINFER_PROFILE_STEPS:-300}"
exec nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --cuda-trace-all-apis=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --sample=none \
  --cpuctxsw=none \
  --kill=none \
  --force-overwrite=true \
  --output="${NSYS_OUT}" \
  "${REAL_WORKER}" "$@" --profile-cuda-steps "${PROFILE_STEPS}"
```

Then `chmod +x /root/RustInfer/result/nsys-shim/rustinfer-worker`.

### 2. Launch the all-in-one server with the shim first in PATH

Tune `--max-batch-seqs` to the target concurrency `c` (e.g. c=32 → `--max-batch-seqs 32`), and set `--max-batch-tokens` and `--mem-fraction-static` large enough. The worker primes CUDA Graphs for decode batches in `[1,2,4,8,16,32]`.

```bash
cd /root/RustInfer
mkdir -p result
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b

RUSTINFER_NSYS_OUT=/root/RustInfer/result/nsys_paged_worker_c32 \
RUSTINFER_PROFILE_STEPS=300 \
PATH=/root/RustInfer/result/nsys-shim:/root/RustInfer/target/release:$PATH \
./target/release/rustinfer-server \
  --port 8014 \
  --model ${MODEL} \
  --model-type llama3 \
  --device cuda:0 \
  --max-batch-tokens 2048 \
  --max-batch-seqs 32 \
  --max-model-len 1024 \
  --kv-cache-mode paged:1 \
  --mem-fraction-static 0.20 \
  --enable-prefix-caching \
  --model-name llama3.2-1b \
  --log-level warn
```

Verify in the log: `CUDA Graphs primed for decode-only batches in [1, 2, 4, 8, 16, 32]` and `Entering serve loop`. Confirm the worker is running under nsys: `pgrep -af nsys` should show `nsys profile ... rustinfer-worker ... --profile-cuda-steps 300`.

### 3. Drive the workload (use `python3`, not `python`)

Match `--concurrency` to `--max-batch-seqs`. Send enough requests that at least `PROFILE_STEPS` decode steps occur before the worker auto-stops.

```bash
cd /root/RustInfer
python3 bench/bench_real_arrival.py \
  --url http://127.0.0.1:8014 \
  --model llama3.2-1b \
  --label RustInfer-PagedPrefix-Llama3.2-1B-nsys-c32 \
  --warmup-requests 8 \
  --num-requests 128 \
  --concurrency 32 \
  --arrival-rate 32 \
  --max-tokens 64 \
  --seed 20260521 \
  --verbose
```

**Expected behavior:** after `PROFILE_STEPS` decode steps the worker calls `cudaProfilerStop()` and exits. The launcher detects the child exit and shuts everything down, so the bench's still-inflight requests will 500/timeout. This is normal — the `.nsys-rep` is already written. Raise `PROFILE_STEPS` (or lower `--num-requests`) if you want the bench to finish cleanly, but capturing the trace is the goal.

### 4. Generate nsys stats

```bash
rm -f /root/RustInfer/result/nsys_paged_worker_c32.sqlite
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum \
  /root/RustInfer/result/nsys_paged_worker_c32.nsys-rep
```

## Delay/Duration Fallback

Use delay/duration only when `--profile-cuda-steps` is unavailable. The shim's `--capture-range=cudaProfilerApi` is preferred because it auto-starts on the first decode step. With delay/duration, set `--delay` long enough to skip model loading and Paged KV allocation, or nsys captures only init kernels (`sin_cos_calc_bf16`) or large H2D copies.

## Operator-Level Sanity Check

Before trusting serving traces, verify nsys can capture CUDA kernels with a deterministic operator test (no launcher involved — wrap the test binary directly):

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

Prioritize `cuda_gpu_kern_sum` for actual Worker GPU bottlenecks. Observed at c=32, llama3.2-1b, 300 decode steps via CUDA Graphs:

- **GEMMs dominate** (`nvjet_sm90_*` cuBLASLt kernels) — roughly 65% of GPU time at high concurrency. Decode is compute-bound on matmuls (QKV/O proj, MLP).
- `flash_paged_decode::paged_decode_pass1_kernel<bf16>` + `paged_decode_combine_kernel` — attention decode, ~9% combined.
- `flash_attn_paged_ragged_kernel<bf16>` — prefill/mixed attention.
- `fused_add_rmsnorm_bf16`, `rope_rotate`, `swiglu_packed`, `scatter_kv_paged` — each small (1–5%).

Treat large `cudaMemcpyAsync` API time carefully: it dominates `cuda_api_sum` (~80%) but mostly reflects synchronization wait behind decode kernels / CUDA Graph launches (`cudaGraphLaunch`), not real copy cost. Confirm against `cuda_gpu_mem_time_sum` (actual H2D/D2H is small) before optimizing copies.

## Cleanup

The launcher usually tears down its children on exit, but verify and force-clean:

```bash
pkill -TERM -f 'rustinfer-(scheduler|worker|server)' 2>/dev/null || true
pkill -TERM -f 'result/nsys-shim' 2>/dev/null || true
sleep 2
pkill -KILL -f 'rustinfer-(scheduler|worker|server)' 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## Keep README in Sync

The same profiling workflow is documented in `/root/RustInfer/README.md` under `Nsight Systems Profile 方法`. Update both this skill and README when profile flags, ports, endpoints, the launcher architecture, or benchmark commands change.
