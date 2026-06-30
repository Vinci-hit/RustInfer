# Handoff: Unified Ragged ABC Pipeline

Status: implemented in eager mode and end-to-end verified.

## Implementation Status (2026-06-29)

Implemented:

- Mixed prefill+decode now enters the ABC eager path through
  `Runtime::step_fused_abc_eager(req, row_kind)`.
- The current input is represented as one flat token tape plus ragged prefix
  metadata from the existing `StepRequest`/`BatchPlan` path.
- `row_kind` distinguishes `Decode`, `PrefillFinal`, `PrefillCont`, and `Pad`.
- CUDA `merge_compact_mixed` commits decode rows and final-prefill rows into
  the next decode A buffer, while ignoring continuation-prefill and pad rows.
- Scheduler mixed fused steps prepare next decode control with the mixed ABC
  sideband output when possible.
- Decode graph capture remains separate work; this change first makes the
  mixed eager flow semantically consistent with ABC.

Verified:

- `cargo check -p infer-worker`
- `cargo test -p infer-worker application::decode_engine::prealloc_tests --lib -- --nocapture`
- `cargo test -p infer-backend-cuda kernels::gather_merge::tests::merge_compact_mixed_respects_row_kind -- --nocapture`
- `cargo build --release -p infer-scheduler -p infer-worker -p infer-server`
- End-to-end OpenAI-compatible streaming test at target `qps=32`: 32 launched,
  32 succeeded, 0 failed, 512 generated output tokens. Worker step traces showed
  real mixed rows with both prefill rows and decode rows in the same fused step.

Known remaining work:

- Mixed CUDA graph capture is not implemented yet.
- Performance is not claimed optimal; qps=32 correctness passed, but performance
  benchmarking was intentionally skipped because GPUs were busy.
- Scheduler can still log KV budget drift warnings because some worker-side
  preallocation is hidden from scheduler accounting; it recalibrates and does
  not block the verified end-to-end flow.
- Full `cargo test -p infer-worker --lib` still hits a pre-existing baseline
  segfault in `models::decoder::tests::component_decoder_ragged_batch_matches_serial`
  and was reproduced on clean HEAD.

## Goal

Replace the current decode-only ABC pipeline with a unified ragged ABC pipeline.

The default worker step should accept one flat token tape plus prefix sums:

```text
A_tokens:  [cap_num_tokens]
cu_q_lens: [cap_rows + 1]
```

Decode-only, prefill-only, and mixed prefill+decode should all use the same logical flow. A pure decode step is just the special case where every logical row has `q_len = 1`.

The CUDA graph work is separate. First make the pipeline semantically uniform and correct in eager mode. Then bucket/pad and capture.

## Current Problem

Current paths are split:

- Pure decode uses the ABC path:
  - `input_ids_buf` as A, shaped `[cap_batch]`
  - `abc.argmax_out_dev` as C
  - `merge_compact_decode`
  - `compact_extend_control`
  - `issue_decode_abc`
- Mixed prefill+decode uses `handle_fused_step -> Runtime::step_fused_eager`.
  - It builds one ragged `StepRequest`.
  - `input_ids_tensor` already flattens all row tokens into `[num_tokens]`.
  - `cu_q_lens` already identifies row ranges.
  - But this bypasses ABC state and runs an eager ragged forward.

This means active decode rows get pulled out of the fast ABC path whenever a prefill arrives. That is the main mixed-batch jitter source.

## Design Principle

ABC should no longer mean "decode batch". ABC should mean:

```text
A = next step input token tape
B = optional admission/staging data
C = sampled row outputs
```

Rows remain necessary. We should avoid thinking in decode/prefill batches, but we still need logical rows because KV tables, stop criteria, output ownership, and active compaction are row/sequence scoped.

Use this terminology:

- `rows`: number of logical sequences/chunks in this step.
- `num_tokens`: total flattened input token count.
- `q_lens[row]`: number of input tokens for that row.
- `cu_q_lens[row]..cu_q_lens[row+1]`: row's token range in `A_tokens`.
- `row_kind[row]`: how to commit the sampled row output.
- `sample_row[row]`: row in hidden/logits to sample, normally `cu_q_lens[row + 1] - 1`.

## Row Kinds

Add a device-side row kind buffer:

```rust
#[repr(i32)]
enum RowKind {
    Decode = 0,
    PrefillFinal = 1,
    PrefillCont = 2,
    Pad = 3,
}
```

Semantics:

- `Decode`: one regular decode row. Sampled token is emitted. If not finished, it becomes next-step decode input.
- `PrefillFinal`: final prefill chunk. Sampled token is emitted as the first decode token. If not finished, it becomes next-step decode input and the sequence enters `active`.
- `PrefillCont`: intermediate prefill chunk. KV is written, but no user token is emitted and no next-step decode row is created.
- `Pad`: inert row for shape padding. No output and no active row.

## New Device Buffers

Extend or replace `AbcBuffers` with ragged-capable buffers:

```rust
pub struct RaggedAbcBuffers<D: Device> {
    // A: flat input token tape for the current/next step.
    pub a_tokens_dev: Tensor<i32, D>,        // [cap_num_tokens]
    pub a_tokens_host: Vec<i32>,             // [cap_num_tokens], pinned later

    // Optional B: upload/staging for suffix admissions or prefill chunks.
    // Stage 1 can skip B and upload directly into a_tokens_dev.
    pub b_tokens_dev: Tensor<i32, D>,        // [cap_num_tokens]
    pub b_tokens_host: Vec<i32>,             // [cap_num_tokens]

    // Row metadata.
    pub row_kind_dev: Tensor<i32, D>,        // [cap_batch]
    pub sample_rows_dev: Tensor<i32, D>,     // [cap_batch]
    pub generated_counts_dev: Tensor<i32, D>,
    pub max_tokens_dev: Tensor<i32, D>,
    pub ignore_eos_dev: Tensor<i32, D>,
    pub eos_ids_dev: Tensor<i32, D>,

    // C: one sampled token per logical row, not per input token.
    pub c_tokens_dev: Tensor<i32, D>,        // [cap_batch]
    pub argmax_ws: Tensor<f32, D>,

    // Mixed merge outputs.
    pub active_src_rows_dev: Tensor<i32, D>,
    pub finished_src_rows_dev: Tensor<i32, D>,
    pub finished_tokens_dev: Tensor<i32, D>,
    pub active_tokens_dev: Tensor<i32, D>,
    pub prefill_final_src_rows_dev: Tensor<i32, D>,
    pub prefill_final_tokens_dev: Tensor<i32, D>,
    pub counts_dev: Tensor<i32, D>,

    // Host mirrors for the sidebands.
    pub counts_host: Vec<i32>,
    pub active_src_rows_host: Vec<i32>,
    pub active_tokens_host: Vec<i32>,
    pub finished_src_rows_host: Vec<i32>,
    pub finished_tokens_host: Vec<i32>,
    pub prefill_final_src_rows_host: Vec<i32>,
    pub prefill_final_tokens_host: Vec<i32>,
}
```

`counts_dev` should include enough fields to let the host commit without reading dynamic device lengths:

```text
counts[0] = active_decode_out
counts[1] = finished_decode_out
counts[2] = prefill_final_out
counts[3] = old_rows
counts[4] = next_decode_rows
```

Exact layout can change, but keep it documented next to the kernel.

## Unified Forward Flow

New runtime entry point:

```rust
Runtime::issue_ragged_abc(
    req: &StepRequest,
    row_kind: &[RowKind],
    a_valid_prefix_tokens: usize,
    reuse_device_control: bool,
    async_next_slots: Option<&[u32]>,
) -> OpResult<()>
```

Initial eager implementation can ignore `a_valid_prefix_tokens` and upload the full token tape every step. Optimize A reuse later.

Flow:

1. Build `BatchPlan` from `StepRequest`.
2. Upload or reuse control:
   - `block_tables`
   - `cu_q_lens`
   - `kv_lens`
   - `seq_positions`
   - `seq_lens_step`
   - `rope_positions`
   - `block2req`
   - `block2tile`
3. Upload `A_tokens` as the flattened `req.seqs[*].input_ids`.
4. Run `run_layers(plan, A_tokens_view)`.
5. Finalize only selected rows:
   - Use `SampleRows::Explicit(&sample_rows)` or a device-side selected-row argmax.
   - Stage 1 can gather selected hidden rows then run lm_head, same as current `LastPerSeq`.
6. Write one sampled token per logical row into `C_tokens`.
7. Run `merge_compact_mixed`.
8. Optionally run `compact_extend_mixed_control` to build next-step pure decode control on device.
9. Async-copy sidebands to host.

Important invariant: `C_tokens[row]` maps to logical row `row`, not token row `cu_q_lens[row + 1] - 1`.

## Mixed Merge Kernel

Replace `merge_compact_decode` with a row-kind-aware kernel:

```c
merge_compact_mixed(
    int* A_tokens_out,
    const int* C_tokens,
    const int* row_kind,
    const int* generated_counts,
    const int* max_tokens,
    const int* ignore_eos,
    const int* eos_ids,
    int eos_len,
    int old_rows,
    int* active_src_rows,
    int* active_tokens,
    int* finished_src_rows,
    int* finished_tokens,
    int* prefill_final_src_rows,
    int* prefill_final_tokens,
    int* counts)
```

Sequential scan is acceptable for stage 1 because `cap_batch` is small.

Pseudo-logic:

```c
active = 0;
finished = 0;
prefill_final = 0;

for row in 0..old_rows {
    kind = row_kind[row];
    token = C_tokens[row];

    if kind == Pad:
        continue;

    if kind == PrefillCont:
        continue; // KV already written by run_layers.

    done = eos_or_max(row, token);

    if kind == Decode {
        if done {
            finished_src_rows[finished] = row;
            finished_tokens[finished] = token;
            finished++;
        } else {
            A_tokens_out[active] = token;
            active_src_rows[active] = row;
            active_tokens[active] = token;
            active++;
        }
        continue;
    }

    if kind == PrefillFinal {
        prefill_final_src_rows[prefill_final] = row;
        prefill_final_tokens[prefill_final] = token;
        prefill_final++;

        if (!done) {
            A_tokens_out[active] = token;
            active_src_rows[active] = row;
            active_tokens[active] = token;
            active++;
        } else {
            finished_src_rows[finished] = row;
            finished_tokens[finished] = token;
            finished++;
        }
    }
}
```

Question for implementation: whether `PrefillFinal` should be reported in a separate sideband or folded into `active/finished`. Keeping it separate makes host commit clearer because prefill final rows need to move from `prefilling`/pending cmd state into `active`.

## Next-Control Kernel

Current `compact_extend_control` assumes every survivor is an old decode row and `q_len=1`.

For the unified pipeline, next step is still pure decode for all active survivors and newly admitted final-prefill rows. Add:

```c
compact_extend_mixed_control(
    block_tables_in,
    block_tables_out,
    kv_lens_in,
    kv_lens_out,
    seq_positions_out,
    seq_lens_step_out,
    rope_positions_out,
    cu_q_lens_out,
    block2req_out,
    block2tile_out,
    active_src_rows,
    counts,
    new_slots,
    mbps,
    cap_batch)
```

It is very close to current `compact_extend_control`:

- `active_src_rows[r]` points to the source row in this mixed step.
- `M = kv_lens_in[src]` is length after the mixed step wrote KV.
- Copy source block table to compacted row.
- Append `new_slots[r]` for the next decode step.
- Set next-step metadata:

```text
kv_lens_out[r]        = M + 1
seq_positions_out[r]  = M
rope_positions_out[r] = M
seq_lens_step_out[r]  = 1
cu_q_lens_out[r + 1]  = r + 1
block2req_out[r]      = r
block2tile_out[r]     = 0
```

This preserves the current async decode control plane after the mixed step. The next issued step is again a normal decode step unless new prefill rows are appended.

## Host Commit Model

The host still owns request lifecycle and maps sequence IDs to state.

Need a `RaggedPendingStep` replacing or extending `PendingDecode`:

```rust
struct PendingRaggedStep {
    rows: Vec<RowMeta>,
    assigned: Vec<AssignedIndices>,
    new_decode_indices: Vec<u32>,
    next_slots: Vec<u32>,
    rows_count: usize,
    device_prepared: bool,
}

struct RowMeta {
    sequence_id: u64,
    kind: RowKind,
    // Index into active row order, prefill plan, or pad sentinel.
    owner: RowOwner,
}
```

Commit rules:

- `Decode` active survivor:
  - append sampled token to existing `ActiveSeq`
  - append this step's allocated KV slot
  - emit `GeneratedToken`
- `Decode` finished:
  - emit token
  - remove active seq
  - free all owned KV if prefix caching disabled
- `PrefillFinal`:
  - reclaim its full block table from prefill plan
  - emit first generated token
  - if not finished, create `ActiveSeq` with sampled token as `last_token`
  - if finished, free or retain according to prefix caching
- `PrefillCont`:
  - update `PrefillSeqMap` with accumulated block table and computed token count
  - no generated token
- `Pad`:
  - free transient scratch/pad blocks

The current `commit_fused_decode` and `commit_prefill_outputs` can be merged into this row-kind commit layer.

## Scheduler / Worker Flow

Current `handle_fused_step` should become a request builder for the unified ABC step, not an eager runner.

New shape:

1. Drain any pending ABC step and send output.
2. Build rows:
   - existing active decode rows first
   - pending prefill chunks next
   - optional pad rows last
3. Allocate KV:
   - one new slot per decode row
   - `new_tokens` slots per prefill chunk
   - optional scratch slots for pad/dummy rows
4. Build `StepRequest` with flattened-capable per-row `SeqStep`s.
5. Build `row_kind`.
6. Call `Runtime::issue_ragged_abc`.
7. Next loop finalizes the pending ragged step.

This keeps the one-step pipeline model:

```text
finalize previous -> issue next -> send previous output
```

Pure decode becomes the same flow with only `Decode` rows and `q_len=1`.

## Migration Plan

Do not rewrite everything in one patch.

### Stage 1: Add Ragged ABC Beside Existing Decode ABC

- Add `row_kind_dev`, `sample_rows_dev`, and `c_tokens_dev`.
- Add `merge_compact_mixed` kernel and Rust wrapper.
- Add `Runtime::issue_ragged_abc`.
- Keep existing `issue_decode_abc` unchanged.
- Route only mixed steps through `issue_ragged_abc`.
- Keep full host uploads. No A reuse optimization yet.

Success criteria:

- Mixed output equals current `step_fused_eager`.
- No persistent corruption after mixed then pure decode.
- No leaks in KV allocator under cancel/finish.

### Stage 2: Unified Commit

- Introduce `PendingRaggedStep` and row-kind commit.
- Replace `commit_fused_decode` + `commit_prefill_outputs` for mixed steps.
- Preserve existing pure decode commit until mixed is stable.

Success criteria:

- Decode rows advance correctly through mixed steps.
- Prefill final rows are admitted to active exactly once.
- Prefill continuation rows produce no user token.

### Stage 3: Move Pure Decode Onto Ragged ABC

- Use `issue_ragged_abc` for decode-only too.
- Keep a fast path inside it for all `q_lens == 1`.
- Delete or demote `merge_compact_decode` once parity is proven.

Success criteria:

- Pure decode throughput does not regress materially.
- ABC A reuse and device-control reuse still work for decode-only.

### Stage 4: Optimize A Reuse

Current A reuse is row-based. Ragged A reuse should be token-prefix-based:

- For decode-only, same behavior as today: surviving rows' next tokens are compacted to `A_tokens[0..active]`.
- For mixed, prefill rows usually require uploading suffix tokens; decode prefix can still be retained.
- Add `a_valid_prefix_tokens` later. Initial implementation can upload all tokens.

### Stage 5: Bucket Graphs

Only after Stage 1-4 are correct.

- Bucket `rows`, `num_tokens`, and `total_q_tiles`.
- Pad rows/tokens with dummy scratch KV.
- Capture `run_layers` first.
- Later capture `finalize selected rows + argmax + merge`.

## CUDA Graph Notes

Ragged control values can be dynamic, but graph-captured shapes cannot be.

To capture mixed ABC:

- `A_tokens` view shape must be bucketed.
- hidden view shape `[bucket_tokens, dim]` must be bucketed.
- GEMM M must be bucketed.
- attention `total_q_tiles` must be bucketed.
- decode-prefix cuDNN batch must be bucketed.

Padding is required:

- pad decode prefix to capture slot
- pad prefill tokens to token bucket
- pad rows to row bucket
- dummy rows write to scratch KV blocks
- real rows must not attend to dummy rows because attention is per-row via `cu_q_lens`

## FA3 Notes

FA3 is not the first fix.

Reasons:

- Current mixed jitter is caused by path split and eager ragged orchestration, not just attention.
- Decode prefix already has a cuDNN paged decode rescue path.
- FA3 would only improve the prefill suffix attention portion.
- It will not fix host control rebuild, ABC bypass, GEMM shape churn, or graph absence.

Use nsys before adding FA3. If `launch_cute_ragged` dominates after unified ABC, then evaluate FA3 for the prefill suffix.

## Invariants To Test

Correctness:

- Pure decode outputs unchanged.
- Single prefill final emits exactly one first token.
- Chunked prefill continuation emits no token until final chunk.
- Mixed decode+prefill output matches current eager fused behavior.
- Mixed step followed by pure decode does not corrupt RoPE/KV control tails.
- EOS/max_tokens works for both decode and prefill-final rows.
- Pad rows produce no output and do not affect real rows.

KV:

- Every allocated decode slot is either committed or freed.
- Prefill continuation KV remains owned by `prefilling`.
- Prefill final KV moves to `active` or is released if finished.
- Prefix caching mode does not prematurely free retained blocks.

Performance:

- QPS=32 mixed workload should show fewer large step-time spikes.
- Pure decode throughput should stay close to current ABC.
- Step trace should show mixed no longer forcing a separate eager `step_fused_eager`.

## Files To Touch First

- `crates/infer-worker/src/application/runtime.rs`
  - new ragged ABC buffers
  - `issue_ragged_abc`
  - finalize/readback for mixed sidebands
- `crates/infer-worker/src/application/decode_engine.rs`
  - `PendingRaggedStep`
  - row-kind commit
  - eventually replace decode-only pending path
- `crates/infer-worker/src/application/worker_scheduler.rs`
  - build row metadata and call ragged ABC instead of `step_fused_eager`
- `crates/infer-backend-cuda/src/kernels/gather_merge/gather_merge.cu`
  - `merge_compact_mixed`
  - `compact_extend_mixed_control`
- `crates/infer-backend-cuda/src/kernels/gather_merge/mod.rs`
  - Rust wrappers for new kernels
- `crates/infer-core/src/plan.rs`
  - optional later: carry precomputed `cu_q_lens/block2req/block2tile` in `BatchPlan`

## First Implementation Cut

The smallest useful cut:

1. Add `RowKind`.
2. Add `row_kind_dev`, `sample_rows_dev`, `c_tokens_dev`.
3. Implement `merge_compact_mixed`.
4. Add `Runtime::step_ragged_abc_eager_finalized` that:
   - uploads full A token tape
   - runs `run_layers`
   - uses `finalize(Explicit(last rows))`
   - argmaxes into `c_tokens_dev`
   - runs `merge_compact_mixed`
   - synchronously reads sidebands
5. Route mixed steps through this function.

This first cut may not yet overlap copy-out or build next control on device. That is acceptable. The first goal is to make the semantics unified and remove the separate `step_fused_eager` concept.

After that, restore async behavior:

- async copy-out
- `compact_extend_mixed_control`
- A token reuse
- graph bucket capture

## Implementation Status

Implemented in the current worktree:

- Added `RaggedRowKind` in `crates/infer-worker/src/application/runtime.rs`.
- Extended `AbcBuffers` with mixed row-kind and prefill-final sideband buffers.
- Added CUDA `merge_compact_mixed` in:
  - `crates/infer-backend-cuda/src/kernels/gather_merge/gather_merge.cu`
  - `crates/infer-backend-cuda/src/kernels/gather_merge/gather_merge.h`
  - `crates/infer-backend-cuda/src/kernels/gather_merge/mod.rs`
- Added `Runtime::step_fused_abc_eager(req, row_kind)`:
  - builds the same ragged `BatchPlan`
  - uploads flat token tape through `prefill_ids_buf`
  - runs eager `run_layers`
  - finalizes `LastPerSeq`
  - writes argmax into ABC C (`abc.argmax_out_dev`)
  - runs `merge_compact_mixed`, writing survivor tokens into decode A (`input_ids_buf`)
  - returns the old row-aligned `StepOutput` contract for scheduler commit
- Updated `handle_fused_step` to call `step_fused_abc_eager` instead of `step_fused_eager`.
- Added scheduler row-kind construction:
  - decode rows -> `Decode`
  - decode-prefix shape pads -> `Pad`
  - final prefill chunks -> `PrefillFinal`
  - continuation chunks -> `PrefillCont`
- Added `DecodeEngine::record_mixed_abc_rows` so a single-group mixed step records the survivor row order whose tokens now live in flat A.
- Added `DecodeEngine::invalidate_abc_reuse` and intentionally invalidated A reuse when one logical mixed round spills into multiple forward groups, because later pure-prefill groups overwrite A with only their own survivors.
- Fixed ABC stop metadata upload to saturate `u32::MAX` to `i32::MAX` instead of wrapping to `-1`.
- Added `Runtime::prepare_mixed_next_decode_control(next_slots)`:
  - reuses the existing decode `compact_extend_control` kernel
  - gathers mixed survivors from `abc.active_src_rows_dev`
  - appends the reserved next-step KV slots
  - leaves `kv_index` ready for the next pure decode step
- Extended `DecodeEngine::record_mixed_abc_rows` to store reserved `next_slots` in `prealloc` and record `device_rows` when the mixed next-control build succeeded.
- Updated `handle_fused_step` to best-effort reserve next-step slots after a successful single-group mixed commit. If reservation or device-control build fails, it falls back to A-token reuse plus host control rebuild.
- Mixed sideband readback now uses the CUDA copy-out stream:
  - downloads `counts`, row-aligned C tokens, `active_src_rows`, and `finished_src_rows`
  - synchronizes copy-out once before returning `StepOutput`
  - avoids the earlier multiple `to_host_vec()` compute-stream synchronizations

Not implemented yet:

- A dedicated `compact_extend_mixed_control` kernel. The current implementation reuses decode `compact_extend_control`, which is sufficient because mixed merge emits compacted next-decode rows via `active_src_rows`.
- 1-deep mixed pipeline. Copy-out is on the copy-out stream now, but the mixed wrapper still waits for it before returning.
- mixed CUDA graph buckets.
- full removal of decode-only ABC; pure decode still uses the old optimized path.

Verification performed:

- `CUDA_VISIBLE_DEVICES=7 cargo check -p infer-worker` passed.
- `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-worker --lib --no-run` passed.
- `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-backend-cuda --lib --no-run` passed.
- Added host-side DecodeEngine state tests for mixed ABC row/prealloc/device_rows bookkeeping. They compile into the `infer-worker` test binary.
- Runtime CUDA libraries are in the conda base environment. Use this prefix when running tests:

```bash
LD_LIBRARY_PATH=/home/liuwenqi/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib:/home/liuwenqi/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/liuwenqi/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

- With that runtime path:
  - `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-worker application::decode_engine::prealloc_tests --lib -- --nocapture` passed: 8 tests.
  - `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-backend-cuda kernels::gather_merge::tests::merge_compact_mixed_respects_row_kind -- --nocapture` passed: 1 test.
  - `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-worker application:: --lib -- --test-threads=1 --nocapture` passed: 29 tests.
  - `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-worker domain:: --lib -- --test-threads=1 --nocapture` passed: 36 tests.
- `cargo test -p infer-worker --lib` still cannot be used as a clean full correctness signal because the baseline HEAD also segfaults in `models::decoder::tests::component_decoder_ragged_batch_matches_serial` under the same environment. This was verified in a clean `git worktree` at HEAD; it is not introduced by the mixed ABC changes.
- No end-to-end mixed prefill+decode test has been run yet. The current evidence covers state bookkeeping and the mixed merge kernel, not full model output equivalence.
