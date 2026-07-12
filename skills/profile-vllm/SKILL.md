---
name: profile-vllm
description: This skill should be used when profiling a vLLM OpenAI server with Nsight Systems as a baseline to compare against RustInfer — isolating the EngineCore (CUDA) process, capturing decode/prefill kernel breakdowns, and running the matched bench_real_arrival workload at the same concurrency (e.g. c=32). Triggers on "profile vLLM", "nsys vLLM baseline", "vLLM kernel breakdown", or "compare vLLM vs RustInfer GPU time".
---

# vLLM Nsight Systems Profiling (RustInfer Baseline)

## Purpose

Profile a vLLM OpenAI-compatible server with `nsys` to produce a GPU-kernel baseline that is directly comparable to the RustInfer worker trace (`skills/rustinfer-nsys-profile`). Use the **same model, same concurrency, and the same `bench/bench_real_arrival.py` workload** so `cuda_gpu_kern_sum` numbers line up side by side.

## Core Rule

Profile the process that runs CUDA. In vLLM v1 that is the **EngineCore** subprocess (`EngineCore pid=...`), not the `APIServer` process. Wrapping the whole `vllm serve` in `nsys` with `--trace-fork-before-exec=true` captures the EngineCore child automatically.

## Environment (this host)

- vLLM venv: `/root/vllm-bench` → vLLM **0.22.0**, Python 3.12. Use `/root/vllm-bench/bin/python` (or activate the venv); do **not** use the system `python`.
- Model: `/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b`, served as model name `llama3.2-1b`.
- vLLM port: **8024** (RustInfer uses 8014 — keep them distinct so both can run / be compared).
- nsys output base: `/root/RustInfer/result/nsys_vllm_worker_c32`.
- bench output JSON: `/root/RustInfer/result/bench_vllm_c32.json`.

## Architecture note (why fork-before-exec)

vLLM v1 splits into:
- `APIServer pid=...` — HTTP/OpenAI front end. No CUDA kernels.
- `EngineCore pid=...` — scheduler + model executor. **All CUDA kernels run here** (GEMMs, FlashInfer attention, CUDA Graphs).

`nsys profile` is launched on the parent `vllm serve` command; `--trace-fork-before-exec=true` makes nsys follow the EngineCore fork so its CUDA activity is captured. There is no separate "worker binary" to wrap like RustInfer — it's one Python process tree.

## Recommended Workflow

### 1. Clean any prior vLLM / nsys processes

```bash
pkill -KILL -f 'vllm|nsys' 2>/dev/null || true
sleep 2
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

### 2. Launch vLLM under nsys

Match `--max-num-seqs` to the target concurrency `c` (c=32 → `--max-num-seqs 32`) so batch shapes align with the RustInfer run. Keep CUDA Graphs **enabled** (do NOT pass `--enforce-eager`) so the trace reflects graph-captured decode, matching RustInfer's primed CUDA Graphs.

```bash
cd /root/RustInfer
mkdir -p result
MODEL=/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b
source /root/vllm-bench/bin/activate

nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --sample=none \
  --cpuctxsw=none \
  --kill=none \
  --force-overwrite=true \
  --output=/root/RustInfer/result/nsys_vllm_worker_c32 \
  vllm serve ${MODEL} \
    --served-model-name llama3.2-1b \
    --port 8024 \
    --max-num-seqs 32 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.20 \
    --dtype bfloat16 \
  > /root/RustInfer/result/vllm_startup_c32.log 2>&1 &
```

### 3. Wait until ready (do not bench early — you'll only capture startup/graph-capture kernels)

Watch the startup log for `Application startup complete.` and the CUDA-graph capture lines (`Capturing CUDA graphs (decode, FULL)`), then health-check:

```bash
for i in $(seq 1 120); do
  curl -s -m5 --noproxy '*' http://127.0.0.1:8024/v1/models | grep -q llama3.2-1b && { echo "vLLM ready (~$((i*2))s)"; break; }
  sleep 2
done
tail -5 /root/RustInfer/result/vllm_startup_c32.log
```

### 4. Drive the matched workload (`python3`, concurrency = max-num-seqs)

Use the **same** `bench/bench_real_arrival.py` and parameters as the RustInfer profile run, only changing `--url` and `--label`:

```bash
cd /root/RustInfer
python3 bench/bench_real_arrival.py \
  --url http://127.0.0.1:8024 \
  --model llama3.2-1b \
  --label vLLM-0.22-Llama3.2-1B-nsys-c32 \
  --warmup-requests 8 \
  --num-requests 128 \
  --concurrency 32 \
  --arrival-rate 32 \
  --max-tokens 64 \
  --seed 20260521 \
  --output-json /root/RustInfer/result/bench_vllm_c32.json \
  --no-plots \
  --verbose
```

### 5. Stop vLLM to flush the trace, then collect stats

Unlike the RustInfer worker (which auto-stops after `--profile-cuda-steps`), vLLM keeps serving. Send SIGINT to the `vllm serve` parent so nsys finalizes the `.nsys-rep`:

```bash
pkill -INT -f 'vllm serve' 2>/dev/null || true
# wait for nsys to finish writing (look for "Generating .../nsys-report" then the .nsys-rep path)
for i in $(seq 1 60); do
  [ -f /root/RustInfer/result/nsys_vllm_worker_c32.nsys-rep ] && pgrep -f 'nsys' >/dev/null || break
  sleep 2
done

rm -f /root/RustInfer/result/nsys_vllm_worker_c32.sqlite
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,osrt_sum \
  /root/RustInfer/result/nsys_vllm_worker_c32.nsys-rep
```

Note: the vLLM `.sqlite` can be large (hundreds of MB) because the full server runs the whole time — this is expected.

## Interpreting Results (vs RustInfer)

Compare `cuda_gpu_kern_sum` against the RustInfer paged-worker trace at the same `c`. Expected vLLM kernel families on Llama-3.2-1B decode:

- **GEMMs dominate** — cuBLASLt `nvjet_*` / cutlass kernels for QKV/O proj and MLP. Same story as RustInfer; this is where most GPU time goes at high concurrency.
- **FlashInfer attention** — `*paged*` / `*BatchDecode*` / `*BatchPrefill*` kernels (vLLM's attention backend), the analogue of RustInfer's `flash_paged_decode::paged_decode_pass1_kernel` + `paged_decode_combine_kernel`.
- **RMSNorm / RoPE / SiLU(SwiGLU) / KV-cache write** — small per-op fused/elementwise kernels, analogous to RustInfer's `fused_add_rmsnorm_bf16`, `rope_rotate`, `swiglu_packed`, `scatter_kv_paged`.

`cuda_api_sum` will again be dominated by `cudaGraphLaunch` / `cudaMemcpyAsync` sync waits — cross-check `cuda_gpu_mem_time_sum` (real H2D/D2H) before concluding copies are a bottleneck, same caveat as the RustInfer skill.

For end-to-end throughput/latency comparison, read the `summary` blocks in `bench_vllm_c32.json` (throughput, tpot_*, lat_p50/p90/p99) against the RustInfer bench JSON at the same concurrency.

## Keep parameters in lockstep with the RustInfer run

For a fair comparison, these MUST match the RustInfer nsys profile run:
- model (`llama3.2-1b`), concurrency (`--max-num-seqs` ↔ `--max-batch-seqs`), `--max-model-len`, `--gpu-memory-utilization` ↔ `--mem-fraction-static`, dtype (bf16).
- bench: same `num-requests`, `concurrency`, `arrival-rate`, `max-tokens`, `seed`, and warmup.
- CUDA Graphs enabled on both sides (no `--enforce-eager`).

If you change a profile/bench parameter, re-run **both** vLLM and RustInfer so the traces stay comparable.

## Cleanup

```bash
pkill -INT  -f 'vllm serve' 2>/dev/null || true
sleep 3
pkill -KILL -f 'vllm|nsys'  2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Trace only has init/graph-capture kernels | Bench started before server ready, or vLLM killed too early | Wait for `Application startup complete.` + CUDA-graph capture lines before benching; stop with SIGINT after the bench |
| `.nsys-rep` never appears / empty | nsys didn't follow the EngineCore fork | Ensure `--trace-fork-before-exec=true` is set |
| GPU OOM at launch | `--gpu-memory-utilization` too high alongside RustInfer | Run one engine at a time, or lower the util; match RustInfer's `--mem-fraction-static` |
| `python: command not found` / wrong vLLM | Used system python | `source /root/vllm-bench/bin/activate` (vLLM 0.22.0) |
| Port already in use | Old vLLM still serving on 8024 | `pkill -INT -f 'vllm serve'` then retry |
