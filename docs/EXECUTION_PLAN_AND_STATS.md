# Execution plans and phase statistics

Ordinary decode, mixed prefill/decode and MTP now execute through
`application::execution::ExecutionPlan::execute`. The plan records the phase,
execution mode, live batch/token shape and workspace contract. It validates the
phase/graph/workspace combination before calling the operation. The operation
receives that validated plan; ABC and mixed graph dispatch use its graph slot/key.
Existing `BatchPlan` remains responsible for tensor indexing, padding and model
shape validation. Graph eligibility still comes from the runtime's graph policy.

This is a shared execution boundary, not a replacement for each algorithm's
orchestrator. Ordinary decode keeps its issue/finalize pipeline. MTP keeps its
synchronous verification transaction and its existing state recovery behavior.
No new MTP graphs or adaptive speculation policy are introduced here.

## Ownership and completion

| Workspace contract | Owner and lifetime |
| --- | --- |
| `Runtime` | Startup staging, hidden and sampling scratch; caller consumes results before reuse |
| `Abc` | Runtime A/B/C buffers; issue enqueues work, matching finalize collects it; existing stream events order overlapping work |
| `Proposer` | Startup token tape, hidden ping-pong and head scratch; tape is borrowed until target verification and catch-up finish |
| `BorrowedTape` | Target borrows the proposer tape during verify/replay; no host re-upload |
| `Recurrent` | Persistent recurrent snapshot; next verification transaction may overwrite it |

Plans carry descriptions of existing borrows, not raw pointers or new allocations.
Rust tensor ownership, shape checks and existing runtime in-flight checks still
establish memory safety. A plan does not independently prove asynchronous lifetime
correctness. Snapshot, restore, replay and readout may only enqueue GPU work;
logical commit remains at the existing successful transaction boundary. Full
attention keeps KV length/lease handling; GDN keeps snapshot/restore/replay.

## Enable statistics

Set these on the worker (or on the launcher so child processes inherit them):

```bash
export RUSTINFER_EXECUTION_STATS_EVERY=64
export RUSTINFER_EXECUTION_GPU_SAMPLE_EVERY=16
export RUST_LOG=info,execution_stats=info
```

`STATS_EVERY=0` or unset disables accounting entirely: no clock reads, metric
locks, timer events or per-phase log records. Enabled counters use fixed storage
allocated at startup. Cloning the metrics handle does not allocate. Logger sinks
may allocate when a record is emitted.

`GPU_SAMPLE_EVERY=0` or unset selects host timing only. With GPU sampling enabled,
CUDA event pairs are allocated at startup, before graph priming and requests.
The CPU backend returns no GPU timers. Sampling records on the compute stream
and polls for completion on a later invocation of the same phase. If a sample is
unfinished, the phase runs normally without overwriting its event pair or waiting.
Host-only Wait/Commit phases never get CUDA timers. Timing failures disable that
phase's GPU timer with a warning; the operation keeps its original error contract.

Every N calls **per owner and phase**, `execution_stats` emits a cumulative
`execution summary`. Every N successfully committed decode rounds it emits
`execution tokens`. Normal owner destruction also logs the final host counters;
forced process termination may leave only the last periodic summary. GPU samples
can lag a phase invocation, and a final pending sample may be absent. Use N=1 for
short diagnostic runs, not for a throughput comparison.

For individual host spans use `RUST_LOG=info,execution_stats=debug`. Existing
`MTP round` debug logs remain available under `infer_worker=debug`.

```bash
python3 scripts/summarize_execution_stats.py path/to/worker.log
python3 scripts/summarize_execution_stats.py path/to/worker.log --json
```

Counters include warm-up and earlier requests in the same worker lifetime. To
isolate a benchmark window, take differences of cumulative calls, host/GPU totals
and token counters at its boundaries; maxima cannot be differenced.

The tool keeps the latest cumulative summary per owner/phase in each file; it
never sums repeated cumulative reports. Use one worker process lifetime per log
file when comparing runs, especially for tensor-parallel workers.

## Reading the numbers

| Phase | Measured region |
| --- | --- |
| Prefill | Request-driven target eager/graph step, including sampling |
| Decode | ABC forward/finalize/argmax submission or ordinary runtime step |
| Mixed | Mixed ABC eager compute/merge region or graph replay; legacy fused eager step |
| Draft | Entire proposer draft call, including metadata, existing wait and token download |
| Verify | Target eager forward and greedy verification, excluding recurrent recovery |
| Snapshot / Restore | Recurrent state copy submission |
| Replay | Retained target prefix forward submission |
| CatchUp | Proposer alignment and head forward, including its existing wait |
| Readout | Target hidden normalization into caller-owned storage |
| Wait | Explicit instrumented completion waits: ABC/mixed copy-out, target recovery/readout, proposer draft/catch-up |
| Commit | Production ordinary ABC/MTP host state and output commit |

`host_total_ms`, `host_mean_ms`, `host_max_ms` are wall-clock call durations.
An asynchronous phase's host time measures submission; a synchronous phase also
includes waits and CPU work. Wait spans can be nested inside Draft/CatchUp. Some
implicit waits, notably token downloads inside greedy verification, remain within
the enclosing phase. **Do not sum phase means or subtract Wait from GPU time.**
Control uploads and scheduling outside these regions are not attributed to them.

`gpu_samples`, `gpu_total_ms`, `gpu_mean_ms` are **sampled compute-stream elapsed
spans**, not sums of kernel runtimes or whole-round latency. They can include
stream waits and host dispatch gaps. Copy-in/out activity on other streams is not
measured separately. A zero sample count means unavailable/not collected, not
zero GPU cost. Compare the sample mean, not sampled total against full host total.

`calls` includes failures; `failures` counts failed operations. Phase `tokens`
counts successfully processed input rows, including speculative and replayed
rows. `graph_calls` counts graph-mode executions (ordinary runtime cold paths may
also warm/capture); it is not a committed-token count.

Token summaries are emitted after successful production ABC/MTP commits and MTP
reference-session rounds. `proposed`/`accepted` count speculative proposals and
prefix matches; `emitted` counts committed output tokens; `materialized` counts
retained input tokens. Bonus tokens are emitted but are not accepted drafts.
Acceptance is weighted as total accepted / total proposed, and is absent when no
drafts were proposed (ordinary decode or K=0). Cancellation can discard issued
work, so processed rows can exceed emitted tokens. Mixed/stochastic serving
commits do not yet contribute token summaries; their execution spans are covered.

## Validation

- CPU execution-plan tests reject verification on ordinary decode graphs and
  invalid workspace use; counters preserve operation failures and disabled mode.
- A mock asynchronous timer verifies unfinished event pairs are never overwritten.
- Hybrid decoder tests exercise K=0..3 and rejection at every position with
  statistics enabled, checking restore/replay counts and greedy continuation.
- CUDA `scope_timing_supports_replay_and_keeps_stream_alive` tests event timing
  around graph replay and persistent timer ownership.
- Python parser tests cover cumulative snapshots, owner separation, ANSI logs and
  absence of an acceptance rate for zero proposals.

This change does not resolve the previously recorded long-text MTP/greedy output
mismatch or stop-string behavior; these remain separate correctness work.

### Local GPU smoke run (2026-09-10)

RTX 4070 Ti SUPER, Qwen3.5-4B BF16; ordinary Graph and MTP K=1, 32-token
requests, one measured repeat. Statistics reported every call, GPU sampled every
8 eligible calls. Artifacts: `target/p1-execution-stats-smoke/`.

Both paths completed. Short greedy comparisons matched 3/3. The benchmark still
exited with its known stop-string failure in both modes; this is not a fully
passing accuracy benchmark or a performance improvement claim.

The MTP worker reported 166 verification rounds and 27 restore/replay operations,
with no phase failures. Cumulative sampled compute-stream means were about
19.57 ms verify, 0.34 ms snapshot, 0.32 ms restore and 15.29 ms replay. These
include warm-up and diagnostic requests. They demonstrate that the new counters
separate state-copy cost from retained-prefix recomputation; use a controlled
measurement window for performance decisions.
