# Graph-prefill via chunked prefill — design + plan

Goal: cut worker prefill from ~7ms eager → ~3-4ms by CUDA-graph-capturing the
prefill forward. Variable prompt length is the blocker; **chunked prefill makes
every prefill step a fixed token count → graph-capturable** (one graph per
token-count bucket, exactly like decode graphs are one-per-batch-size).

## De-risk (DONE, 2026-06-25)
- Ragged prefill attention IS graph-capturable: `flash_attn_paged_prefill.cu`
  launches a single `kernel<<<grid(total_q_tiles, num_q_heads), block>>>` with
  NO cudaMalloc/memset/sync/memcpy. Grid is fixed for a fixed bucket. kv_lens /
  cu_q_lens / block_tables are device ptrs read at runtime (variable kv_len is
  fine, like decode).
- Decode graph machinery to mirror: `runtime.rs` `decide()` →
  `GraphDecision::{Eager,Graph(slot)}`; `step_graph()` cold path = eager warmup
  (populate cuDNN/cuBLASLt shape caches that are capture-illegal) → synchronize →
  `graph_capture_begin` → capture → `graph_capture_end(key)` → `graph_launch`.
  All input buffers are fixed addresses (`input_ids_buf`, `kv_index.*`, `hidden`,
  `kv_pool`, `abc.argmax_out_dev`); metadata uploaded each step in `upload_index`.

## Measured baseline (this session)
TTFT short prompt ~12.6ms = worker ~6.5-7.2ms (eager forward) + IPC ~3.7ms
(park-wake) + client ~1.5ms. lm_head last-token + build-free eager GEMM already
landed (60ms→12.6ms). Worker eager-forward is the target here.

## Design
- **Buckets** (prefill capture sizes, analogous to decode capture_sizes):
  e.g. `[32, 64, 128, 256]`. `chunked_prefill_size = 256` (max bucket) so the
  scheduler splits prompts into ≤256-tok chunks (machinery already exists,
  `continuous_batching.rs`, currently disabled via `chunked_prefill_size=0`).
- **Capture region = `run_layers` only** (embed + decode_layers → writes KV).
  finalize+sample stays EAGER and only on the last chunk (it's `LastPerSeq`,
  M=batch, already cheap). So the graph holds just the 36-layer forward.
- **Padding** (the hard part): a chunk has `actual` ≤ bucket tokens. The graph is
  fixed at `bucket` tokens, so GEMMs compute `bucket` rows and attention launches
  `ceil(bucket/128)` Q-tiles. Pad with a **dummy tail seq** of `bucket-actual`
  tokens that writes KV to a dedicated **scratch block range** (never read by
  real seqs). cu_q_lens=[…, real_end, bucket], block2req/block2tile include the
  dummy. Causal attention + cu_q_lens ⇒ real tokens never attend to dummy; dummy
  output rows are discarded. Real seq's KV/logits byte-identical to eager.
  - Scratch KV: preallocate `ceil(max_bucket/block_size)` dummy blocks once.
- **decide()**: ragged plan, round `num_tokens` up to nearest bucket ≤ max; if a
  graph slot exists → new `GraphDecision::PrefillGraph(bucket)`. Keep decode keys
  and prefill keys in separate slot namespaces (or a tagged key) so they don't
  collide.
- **prefill-graph step** (mirror step_graph): build padded plan → upload metadata
  (input_ids, cu_q_lens, kv_lens, block_tables, block2req, block2tile,
  rope_positions) into fixed buffers → replay-or-(warmup+capture) `run_layers` →
  if last chunk: eager `finalize(LastPerSeq)` + sample → first token.

## Status (2026-06-25)
- **STAGE A DONE — single-seq prefill graph (exact `num_tokens` key), no
  padding. Landed in `runtime.rs`.** `GraphDecision::PrefillGraph(num_tokens)`;
  `GraphRunner::decide` routes single-seq (`batch==1`) plain-`Ragged` prefill of
  `2..=PREFILL_GRAPH_MAX_TOKENS` (256) tokens to a prefill graph keyed by exact
  `num_tokens` (tagged `1<<40` so it can't collide with decode batch keys).
  `step_prefill_graph` captures `run_layers` ONLY (warmup eager pass → sync →
  capture → replay), `sample_tail` stays eager. Budget `PREFILL_GRAPH_BUDGET=16`
  distinct lengths, then eager. Crucially it does NOT set `prefill_gemm_mode`, so
  the warmup builds the capturable per-`(M,N,K)` cuBLASLt cache (the eager `(N,K)`
  path is capture-illegal). Validated: Paris/Tokyo/2+2 correct; captures fire
  (num_tokens=13/20/21/23/27); replay confirmed.
  - **RESULT (H200 cuda:7, seq2.py keep-alive, NW=3, 3×150, back-to-back):**
    HEAD median **15.3ms** vs `1cd848` baseline **16.5ms** — 1.2ms (7.3%) faster,
    no median overlap, p90 15.8 vs 16.7. Worker `handle_prefill` 8.18→**7.48ms**
    (the −0.7ms is the whole win; prefill is GPU-bound — a serial 36-layer chain
    of ~144 latency-bound small-M GEMMs + 36 attn, so graph replay only removes
    the non-overlapped launch slice, NOT kernel-execution latency). Worker
    prep/send is negligible (~0.06ms); qkv+gate_up already fused (4 GEMM/layer).
  - **STAGE B (still TODO) = bucketing + dummy-tail padding** (below) to (a) make
    ANY prompt length capturable with a bounded graph set instead of per-exact,
    and (b) cover burst (batch>1) prefill. Stage A already wins the short-prompt
    TTFT target; Stage B generalizes + bounds graph memory. Going materially
    below 7.48ms needs kernel-level work (fused MLP/attention, larger GEMM tiles)
    — out of scope for the graph capture.
- STEP 1 DONE — chunked-prefill foundation enabled + VALIDATED. Set
  `chunked_prefill_size = 256` in rustinfer.toml. 997-tok prompt → 4 chunks
  (worker logged 4 handle_prefill calls, 23/33/43/51ms as KV prefix grows),
  output correct ("capital of Italy is Rome"); short prompts stay 1 chunk
  (Paris ✓). KV accumulation across chunks is correct. So the scheduler/worker
  chunking path works — graph capture can build on it.
- NOTE: chunking-alone REGRESSES long-prompt TTFT (more eager steps) until the
  graph capture below lands. Short-prompt TTFT (the <8ms target) is unaffected
  (1 chunk). REVERTED to `chunked_prefill_size = 0` for the clean checkpoint —
  **re-enable 256 as the first action when resuming the capture work.**
- Env-gated diagnostics left in tree (off by default, useful for the next push):
  `RUSTINFER_GEMM_BUILD_TRACE` (matmul.cu per-shape build timing),
  `RUSTINFER_TTFT_TRACE` (worker handle_prefill wall, serve_loop.rs),
  `RUSTINFER_SCHED_TRACE` (scheduler dispatch/forward µs, event_loop.rs).
  Server first-token timing at `debug` (chat.rs/streaming.rs TTFT_TRACE).
- REMAINING: steps 2-5 below (the graph capture + padding — the actual perf win).

## Build order
1. DONE (above). Config `chunked_prefill_size` + (next) bucket list + scratch KV.
2. Worker `decide()` + `GraphDecision::PrefillGraph` + bucket rounding.
3. Padded-plan builder (inject dummy tail seq + scratch blocks into cu_q_lens /
   block2req / block2tile).
4. prefill-graph step: capture/replay run_layers; eager finalize+sample on last
   chunk; intermediate chunks no output.
5. Validate: Paris/Berlin/2+2 correctness + churn determinism; measure TTFT
   (expect worker ~7→~4ms) and long-prompt (chunked) throughput.

## Risks / watch
- Padding KV isolation (dummy must never be attended / must not overwrite real
  blocks). Test with churn + canary.
- Graph key namespace collision decode↔prefill.
- cuDNN is decode-only; prefill uses CUTE ragged — capture warms cuBLASLt algo
  cache (eager warmup pass handles it); confirm the (N,K,M-bucket) eager GEMM
  cache plays with capture (under capture it must hit the decode cuBLASLt cache,
  NOT the eager build path — the prefill-graph eager-warmup must populate the
  capturable cache for the bucket M, and `set_prefill_gemm_mode` must be OFF
  during prefill-graph capture so it uses the capturable cuBLASLt path, not
  chunked/eager). IMPORTANT interaction with this session's eager-GEMM flag.
- Intermediate-chunk "no output" path + scheduler continuation semantics.
