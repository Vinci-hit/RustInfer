# Handoff: cuDNN + orchestration regression fix (RustInfer decode)

Date: 2026-06-25. Branch: `feat/worker-batch-forward` (HEAD = `cac326a`).
GPU: H200 `cuda:7`. Model: Qwen3-4B at `/mnt/md2/liuwenqi/vllm_bench/dir` (served id = `dir`).

## TL;DR

The modular refactor after commit **`96f7b4e`** ("repair high qps some seq repeat", the
last known-good + highest-perf commit) regressed the decode path on **two independent
axes**, both now fixed in the working tree (uncommitted):

1. **Perf + part of correctness** — refactor disabled cuDNN decode attention and fell back
   to a custom split-KV flash-decode kernel that is **3.3× slower** AND buggy.
   Fix: restore cuDNN-in-graph attention.
2. **Correctness** — the forward/decode **orchestration** rewrite (commits da7a2b6..cac326a)
   produced wrong tokens (induction/copy collapse: "France is France is", "\n\n\n").
   Fix: revert `runtime.rs` + `decode_engine.rs` to `709eecd` (last-good).

After both fixes: **fresh requests are byte-identical-correct vs 96f7b4e**, perf +35% over
broken HEAD (still ~20% short of 96f7b4e — see Remaining work).

## Root cause (full chain, evidence-backed)

- All CUDA **kernels (.cu) are byte-identical** between 96f7b4e and HEAD. The regressions
  are in Rust **dispatch + orchestration**, not the operators.
- nsys (batch32, 400 steps, both fully CUDA-graphed):
  - 96f7b4e decode attn = **cuDNN SDPA, 14.4µs/layer, 207ms** (36 cuDNN calls captured per
    graph launch — **cuDNN IS graph-capturable**, earlier "can't be graphed" claim was wrong).
  - HEAD decode attn = custom `paged_decode_pass1` **43.9µs** + `paged_decode_combine` 4.4µs
    = **694ms → 3.3× slower**. HEAD also has 1.64× more kernel launches (smaller GEMM tiles +
    `nvjet_..._badd` + `hgemv`) → the remaining perf gap.
- HEAD's `cudnn_paged_attention.cu` had a `stream_is_capturing(stream) → decline` guard
  (added post-96f7b4e) that forced the slow+buggy custom kernel under graph capture.
- Correctness: prefill + sampling are actually correct (verified: `kernel_tok==host_argmax`,
  token 12095=" Paris", logit 16.5). The induction/copy collapse came from the
  `runtime.rs`/`decode_engine.rs` orchestration rewrite (bisect: GOOD at 709eecd, BAD at HEAD;
  intermediate commits da7a2b6/874eb13/... don't build individually = WIP).

## Changes made (working tree, UNCOMMITTED)

| File | Change | Keep? |
|---|---|---|
| `crates/infer-backend-cuda/src/kernels/flash_attn_gqa/cudnn_paged_attention.cu` | Removed the `stream_is_capturing → CUDNN_STATUS_NOT_SUPPORTED` hard-decline (kept the cached-plan-during-capture path) | **YES** |
| `crates/infer-backend-cuda/src/kernels/flash_attn_gqa/mod.rs` | Decode default → cuDNN: `let use_cudnn = std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_none();` | **YES** |
| `crates/infer-worker/src/application/runtime.rs` | Reverted to `709eecd` + 3 API adaptations: `ForwardScratch::new(.., cb)`, two `argmax_into(.., None)` | **YES** |
| `crates/infer-worker/src/application/decode_engine.rs` | Reverted to `709eecd` | **YES** |
| `crates/infer-backend-cuda/src/kernels/flash_attn_gqa/flash_attn_paged_decode.cu` | Adaptive split-cap (`compute_splits_cap`, skip combine when cap==1) — optimizes the custom fallback (now DORMANT since cuDNN is default). Harmless. | optional |
| `crates/infer-worker/src/models/decoder.rs` | `RI_DEBUG_PREFILL` stream dump (DBG-FWD) — **debug only, remove before commit** | **NO (remove)** |

Recreate the orchestration revert from scratch if needed:
```
git checkout 709eecd -- crates/infer-worker/src/application/runtime.rs \
                        crates/infer-worker/src/application/decode_engine.rs
# then re-add 3 API args: ForwardScratch::new(device,dims,cap_num_tokens, cb)
# and argmax_into(&ctx,&logits.0,&mut out,&ws, None) at both call sites
```

## How to build & run (IMPORTANT gotchas)

- **Build to /home, not /mnt/md2** — md2 had been 100% full (now ~217G free, but keep using
  md0 to be safe): `export CARGO_TARGET_DIR=/home/liuwenqi/ri_target`
- **CUDA libs:** `export LD_LIBRARY_PATH=/home/liuwenqi/miniconda3/lib:$LD_LIBRARY_PATH`
- **Build:** `cargo build --release -p infer-worker -p infer-scheduler -p infer-server`
  (pre-modular-split commits like 96f7b4e have 2 worker bins → use `--bin rustinfer-worker`).
- **Launch each daemon as its OWN background task** (separate Bash calls). `& disown` inside one
  call gets SIGTERM'd. Order: scheduler → worker → server. Binaries in
  `/home/liuwenqi/ri_target/release/{rustinfer-scheduler,rustinfer-worker,rustinfer-server}`.
- HEAD `rustinfer.toml` already = `cuda:7` + correct model (no `--config` needed).
  For 96f7b4e use `--config <scratchpad>/ri_96f7b4e.toml` (its toml points to cuda:0 + /root model).
- Worker ready marker in log: `Entering serve loop`. Graph capture for all `capture_sizes` runs
  at startup; cuDNN plan must build at warmup so capture can reuse the cached plan.

## Test commands

Correctness (raw bypasses chat template — cleanest):
```
curl -s localhost:8000/v1/completions -d '{"model":"dir","prompt":"The capital of France is","temperature":0.0,"max_tokens":8}'
# expect: " Paris. The capital of Germany is Berlin"
```
Chat: `{"model":"dir","messages":[{"role":"user","content":"What is 2+2?"}],...}` → "2 + 2 = 4."
Perf: `scratchpad/measure.py <batch> 200` (ITL + agg tok/s). Batched determinism: `scratchpad/verify2.py`.
**Use a freshly-restarted server** for clean correctness — KV recycling contaminates later requests.

## Results

measure.py `<batch> 200` (matches original methodology; aggregate steady-state tok/s):

| | broken HEAD | cuDNN-only fix | **+ GEMM fix (now)** | 96f7b4e |
|---|--:|--:|--:|--:|
| correctness (fresh, single + batched) | ✗ copy/garbage | ✓ correct | **✓ correct** | ✓ correct |
| b32 tok/s | 5863 | 7138 | **8759 (+3.2%)** | 8484 |
| b64 | 9624 | 12853 | **16407 (+8.2%)** | 15164 |
| b128 | 15182 | 20566 | **25721–25924 (≈99%)** | 26217 |
| b256 | — | — | **50560 (≈97%)** | ~52000 |

Fresh 32-concurrent of one prompt → 32/32 identical correct (verify2.py). Goal met: b32/b64
now **exceed** 96f7b4e; b128/b256 match within run-to-run noise.

## GEMM fix — root cause CORRECTION + what was done (2026-06-25, session 2)

The earlier claim "all .cu byte-identical between 96f7b4e and HEAD" was WRONG for `matmul.cu`.
The bisect missed it because the refactor **moved the file path**
(`infer-worker/src/infrastructure/cuda/kernels/matmul/matmul.cu` → `infer-backend-cuda/src/kernels/matmul/matmul.cu`),
so `git diff 96f7b4e HEAD -- <new path>` showed it as a pure addition, not a rewrite.

What actually changed in `gemm_cublasLt_AxBT_RowMajor_bf16` (the bf16 GEMM used by decode at M=batch≥2):
- **96f7b4e**: direct `cublasLtMatmul(algo=nullptr)` → cuBLASLt runtime picks a large-tile,
  single-pass kernel. Fast, and graph-captured fine.
- **broken HEAD**: rewritten to legacy `cublasGemmEx` + **K-chunking@2048 with beta=1 accumulation**
  (to dodge a perceived "split-K not graph-capturable" problem). For down_proj (K≈9728) that is
  5 chunked GEMMs → the `nvjet_..._badd` small-tile launches nsys saw. ~1.6× launches, ~20% slower.

Fix (committed in `matmul.cu`): `gemm_cublasLt_AxBT_RowMajor_bf16` now uses the **per-shape
benchmarked + capturability-filtered cuBLASLt algo cache** that HEAD already contained but left as
dead code (`zimage_build_bf16_gemm_entry`, lines ~233-403). Path:
- eager warmup forward (runtime.rs:597-605, runs before every graph capture) → cache miss & not
  capturing → build entry: heuristic(32 candidates) → bench only the ones that pass a
  `cudaStreamBeginCapture` probe → keep fastest capturable algo, cache per (M,N,K).
- under capture → cache hit → `cublasLtMatmul(cached algo)`. cuBLASLt takes stream+workspace as
  call args (no `cublasSetStream`/`SetWorkspace`), so it is capture-safe with no handle reconfig.
- fallback `gemm_bf16_chunked_legacy` kept ONLY for a cold shape first seen under capture (never
  fires in steady state). Workspace = 4 GiB (config.rs `DEFAULT_GEMM_WORKSPACE_SIZE`) → plenty of
  large-tile algos eligible.

This BEATS 96f7b4e at small batch because we benchmark+pick the fastest *capturable* algo per shape
rather than relying on cuBLASLt's `nullptr` default heuristic.

## Remaining work (pick up here)

1. ~~GEMM tiling alignment~~ **DONE** (see above). b32/b64 now exceed 96f7b4e; b128/b256 within noise.
   Any further large-batch gain is attention-dominated (longer KV at maxtok=300 widens it for BOTH
   builds equally) — diminishing returns; would need fresh nsys vs a rebuilt 96f7b4e on this host.
2. ~~KV-recycle contamination~~ **FIXED (2026-06-25, session 2).** Was NOT KV-block recycling —
   root cause = **stale per-sequence control-buffer tails**. The persistent device buffers
   `cu_q_lens` / `seq_lens_step` / `seq_positions` / `kv_lens` (runtime.rs `KvIndexTensors`) are
   sized to `cap_batch` (256) but `upload_index` only prefix-filled them (`upload_i32_prefix`).
   The attention/scatter kernels iterate over `seq_positions.shape()[0]` (== capacity), so after
   one large batch the tail keeps stale values; a later smaller batch leaves phantom rows whose
   stale `cu_q_lens` make them **claim real tokens and re-apply RoPE in-place at the wrong
   position**, corrupting Q/K. Symptom: one high-batch step permanently poisons every later
   request (fluent but input-incoherent: "capital of France"→"a 1984 American science fiction
   film", 2+2→"10000000"); model weights/RoPE/embeddings all verified byte-intact (it's the
   per-step inputs, not the params). FIX: `upload_i32_full_zeropad` zero-pads those four control
   buffers to full capacity each step so phantom rows are inert (matches the known-good clean
   state). Same *class* as the old `qkv_norm_rope_scatter` capacity-vs-actual OOB, different
   buffers + consumer. Verified: 40-wave × 48-concurrent churn → 0/40 corrupt (was 40/40 by wave
   ~13); verify2 b1==b32 now true for all prompts; perf unchanged. Full root-cause method:
   checksum-probe bisection (params intact → hidden identical at embed, diverges at L0 → dumped
   L0 kernel inputs → saw stale tail). See memory `kv-stale-control-buffer-tail`.
3. **Commit the fix.** cuDNN restore + orchestration revert + **GEMM cache wiring** +
   **stale-tail zero-pad** (runtime.rs). All debug probes removed (decoder.rs / qkv mod.rs
   reverted clean; only runtime.rs + matmul.cu + flash_attn_gqa changes remain). Dormant
   adaptive-split in `flash_attn_paged_decode.cu` is harmless — keep or drop.

## Key artifacts (scratchpad: /tmp/claude-1011/.../7181ccd7-.../scratchpad/)

- `measure.py` (ITL/throughput), `verify2.py` (batched determinism), `ri_96f7b4e.toml` (cuda:7 config)
- nsys traces: `ri96_b32.{nsys-rep,sqlite}` (96f7b4e) vs `result/RI_concurrent32.sqlite` (HEAD); `cmp_nsys.py`
- nsys binary: `/tmp/nsight-systems-2024.6.2/opt/nvidia/nsight-systems/2024.6.2/bin/nsys`
