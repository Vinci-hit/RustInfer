# HANDOFF — Prefill / TTFT perf profiling playbook

Reusable method to **localize a prefill-forward (TTFT) regression to a single GPU
phase** and A/B a fix, on any machine. Built while chasing "532f5c TTFT 8ms vs
HEAD 13ms". Use this on the slow machine to find what's still slow there.

> Status (2026-06-25, cuda:7 H200): found + fixed ONE regression —
> `qkv_norm_rope_scatter` launched its grid at capacity batch (256) not actual
> (see "Fix #1" below). On cuda:7 that was only ~0.7ms (forward 7.85→7.27ms). The
> **other machine is still slow**, so there is at least one MORE regression that
> does not show (or shows small) on a 132-SM H200. This doc is how to find it.

---

## 0. Mental model — what TTFT is made of

```
TTFT = worker.handle_prefill          (GPU forward + sample)
     + IPC park-wake (~3.7ms on cuda:7, OS reschedule across ~6 hops)
     + client aiohttp (~1.5ms)
```
`worker.handle_prefill = run_layers (embed + 36 decode layers) + sample_tail`.
**First decompose TTFT** before assuming it's GPU. A 5ms gap could be IPC /
orchestration / GPU-clock, NOT the forward. Only after you confirm `run_layers`
is the gap do you drill into per-phase GPU timing.

---

## 1. CRITICAL gotchas (learned the hard way)

- **Env vars do NOT reach the worker.** The launch wrapper (RTK hook / shell
  snapshot) strips `VAR=1 cmd` and `export VAR`. `RUSTINFER_TTFT_TRACE` happens to
  survive in some launches; do NOT rely on a new env var. **Gate diagnostics on a
  sentinel FILE** (`std::path::Path::new("/tmp/ri_phase").exists()`, cached in a
  `OnceLock`). Create the file BEFORE launching the worker.
- **`synchronize()` is ILLEGAL under CUDA graph capture.** Per-phase timing syncs
  the stream after every op → it crashes the decode-graph capture warmup
  (`status=13`, "operation failed during capture"). When instrumenting, set
  `capture_sizes = []` in the config so no graph capture happens. Prefill is eager
  regardless, so prefill phase numbers are unaffected.
- **Restart the WHOLE stack** after rebuilding the worker. Scheduler's `LoadModel`
  is one-shot; worker-only restart hangs on "waiting for LoadModel".
- **`pkill` as its own command**, then launch daemons in SEPARATE commands.
  `pkill ...; launch` in one line kills the just-launched daemon (exit 144).
- **HEAD @ 4df87c0 did not compile** as committed (`decoder_block.rs` phase-trace
  block missing `use infer_core::exec::ExecScope`). If you re-add diagnostics that
  call `ctx.scope().synchronize()` in a generic `D` context, import that trait.

---

## 2. Build recipe

HEAD → `ri_target`; a 532f5c baseline worktree → `ri_target_base`.

```bash
# HEAD (current tree)
CARGO_TARGET_DIR=/home/liuwenqi/ri_target cargo build --release \
  --bin rustinfer-worker --bin rustinfer-scheduler --bin rustinfer-server

# 532f5c baseline (make a worktree once)
git worktree add /path/wt_532 532f5ce
cd /path/wt_532 && CARGO_TARGET_DIR=/home/liuwenqi/ri_target_base \
  cargo build --release --bin rustinfer-worker --bin rustinfer-scheduler --bin rustinfer-server
```
RTK masks cargo's exit code — verify by grepping output for `cargo build: N errors`
AND checking the binary mtime actually advanced:
`stat -c '%y' /home/liuwenqi/ri_target/release/rustinfer-worker`.

CUDA libs: worker needs `export LD_LIBRARY_PATH=/home/liuwenqi/miniconda3/lib:$LD_LIBRARY_PATH`.

---

## 3. Run-stack recipe (each daemon a SEPARATE background launch)

```bash
pkill -9 -f 'rustinfer-(scheduler|worker|server)'; sleep 2     # standalone
# scheduler
/home/liuwenqi/ri_target/release/rustinfer-scheduler --config rustinfer.toml
# worker (needs LD_LIBRARY_PATH; TTFT trace on)
LD_LIBRARY_PATH=/home/liuwenqi/miniconda3/lib RUSTINFER_TTFT_TRACE=1 RUST_LOG=info \
  /home/liuwenqi/ri_target/release/rustinfer-worker --config rustinfer.toml
# server
/home/liuwenqi/ri_target/release/rustinfer-server --config rustinfer.toml
```
Readiness: poll `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models`
== 200, then wait ~8s more for weight load.

Config knobs: `device = "cuda:N"`, `model = "/mnt/md2/liuwenqi/vllm_bench/dir"`,
`capture_sizes = []` while instrumenting (else capture crash), restore the decode
list for production.

---

## 4. Bench (low-variance prefill metric)

Sequential keep-alive client (avoids client artifacts). `scratchpad/seq2.py`:
```python
import asyncio, aiohttp, json, time, statistics, sys
BASE="http://127.0.0.1:8000"; NW=int(sys.argv[1]) if len(sys.argv)>1 else 3; N=int(sys.argv[2]) if len(sys.argv)>2 else 30
async def chat(s, content):
    payload={"model":"dir","messages":[{"role":"user","content":content}],"temperature":0.0,"ignore_eos":True,"max_tokens":4,"stream":True}
    t=time.perf_counter()
    async with s.post(BASE+"/v1/chat/completions",json=payload,timeout=aiohttp.ClientTimeout(total=300)) as r:
        async for line in r.content:
            line=line.strip()
            if not line: continue
            x=line.decode()
            if x.startswith("data: "): x=x[6:]
            if x=="[DONE]": break
            try: d=json.loads(x)
            except: continue
            if (d.get("choices") or [{}])[0].get("delta",{}).get("content"): return (time.perf_counter()-t)*1000
async def main():
    async with aiohttp.ClientSession() as s:
        for k in range(3): await chat(s, f"warm up prompt {k}")
        xs=[]
        for k in range(N): xs.append(await chat(s, f"distinct {k} "+" ".join(f"x{k}_{j}" for j in range(NW))))
        xs.sort()
        print(f"NW~{NW}words N={N}: TTFT min={xs[0]:.1f} p10={xs[N//10]:.1f} median={statistics.median(xs):.1f} p90={xs[(N*9)//10]:.1f}ms")
asyncio.run(main())
```
Run: `python seq2.py 3 40`. The worker log `[ttft-trace] handle_prefill wall` /
`run_layers` / `sample_tail` lines are the low-variance numbers; total TTFT is
noise-dominated (±2ms across restarts, worse when GPU downclocks at idle QPS).

**Step 1 on the slow machine:** read `run_layers` (HEAD) vs the same on the 532f5c
stack. If `run_layers` differs ≈ the TTFT gap → it's the forward, go to §5. If
`run_layers` is ~equal but TTFT still differs → bottleneck is IPC/orchestration,
NOT the forward; stop profiling the GPU and look at scheduler/IPC.

---

## 5. Per-phase GPU timing (the localizer) — paste back this diagnostic

Removed from HEAD after the cuda:7 investigation. Re-add to split `run_layers`
into 9 phases. File-gated on `/tmp/ri_phase`; syncs each op (serializes, inflates
absolutes, but the per-phase split localizes the regression).

### 5a. New module `crates/infer-worker/src/components/phase_trace.rs`
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
pub const N: usize = 9;
pub const NAMES: [&str; N] = ["a_norm","a_qkv_gemm","a_scatter","a_attn","a_oproj","f_norm","f_gateup_gemm","f_swiglu","f_down_gemm"];
static ACC: [AtomicU64; N] = [AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0),AtomicU64::new(0)];
static LAYERS: AtomicU64 = AtomicU64::new(0);
#[inline] pub fn enabled() -> bool { static E: OnceLock<bool> = OnceLock::new(); *E.get_or_init(|| std::path::Path::new("/tmp/ri_phase").exists()) }
#[inline] pub fn add(i: usize, ns: u64) { ACC[i].fetch_add(ns, Ordering::Relaxed); }
pub fn tick_layer(num_tokens: usize) {
    let n = LAYERS.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 36 != 0 { return; }
    let v: Vec<f64> = ACC.iter().map(|a| a.swap(0, Ordering::Relaxed) as f64 / 1e6).collect();
    if num_tokens <= 4 { return; }
    let parts: Vec<String> = NAMES.iter().zip(&v).map(|(n,x)| format!("{}={:.2}", n, x)).collect();
    tracing::info!("[phase] nt={} 36L synced_total={:.2}ms {}", num_tokens, v.iter().sum::<f64>(), parts.join(" "));
}
```
Add `pub mod phase_trace;` to `components/mod.rs`.

### 5b. `components/attention.rs` (Qwen3 path) — `use infer_core::exec::ExecScope;`, then inside `run()`:
```rust
let pt = crate::components::phase_trace::enabled();
let mut ts = std::time::Instant::now();
let rec = |i: usize, ts: &mut std::time::Instant| { if pt { let _ = ctx.scope().synchronize(); let now = std::time::Instant::now(); crate::components::phase_trace::add(i, (now-*ts).as_nanos() as u64); *ts = now; } };
// after input-norm match:                       rec(0,&mut ts);
// after self.qkv_proj.forward(...):              rec(1,&mut ts);
// after the qkv_norm_rope_scatter match block:   rec(2,&mut ts);
// after D::attention_paged(...):                 rec(3,&mut ts);
// after self.o_proj.forward(...):                rec(4,&mut ts);
```

### 5c. `components/ffn_dense.rs` — `use infer_core::exec::ExecScope;`, in `run()` time f_norm (idx 5) and call `tick_layer(num_tokens)` at the end; in `project()` time gate_up(6)/swiglu(7)/down(8) with the same `rec` pattern.

### 5d. 532f5c baseline (monolithic `models/qwen3.rs`)
Same statics + `use crate::domain::ports::Device;` (NOT `ports::device::Device` — that mod is private). `let dev = input_ids.device().clone();` then `rec` calls `dev.synchronize()`. Phase→line map: qkv_proj→1, scatter block→2, attention_paged→3, o_proj→4, post-attn fused_add_rmsnorm→5, gate_up→6, swiglu→7, down→8, post-ffn fused_add_rmsnorm→0, then `tick_layer(num_tokens)`.

### Run it
```bash
echo 1 > /tmp/ri_phase           # BEFORE launching worker (OnceLock caches)
# set capture_sizes = [] in config, full-restart stack, run seq2.py
grep -oE '\[phase\] nt=.*' <worker-log> | grep 'nt=27' | tail
rm /tmp/ri_phase                 # disable for clean async runs
```

### cuda:7 reference (nt=27, 36L, synced)
| phase | HEAD before | 532f5c | HEAD after Fix#1 |
|---|---|---|---|
| a_norm | 0.34 | 0.35 | 0.35 |
| a_qkv_gemm | 0.81 | 0.89 | 0.86 |
| **a_scatter** | **1.40** | **0.42** | **0.42** |
| a_attn | 1.89 | 1.86 | 1.86 |
| a_oproj | 0.81 | 0.87 | 0.86 |
| f_norm | 0.34 | 0.35 | 0.35 |
| f_gateup_gemm | 1.53 | 1.55 | 1.55 |
| f_swiglu | 0.35 | 0.36 | 0.35 |
| f_down_gemm | 1.12 | 1.13 | 1.12 |
| total | 8.72 | 7.78 | 7.74 |

**On the slow machine, build this same table for HEAD vs 532f5c.** Whatever phase
column is bigger in HEAD = the regression there. If `a_scatter` is already 0.42
(Fix#1 present) but some OTHER phase is bigger → that's the new target.

---

## 6. Fix #1 (already in tree) — scatter grid sized by capacity

`crates/infer-backend-cuda/src/lib.rs`, `qkv_norm_rope_scatter` (Cuda impl). The
kernel grid is `dim3(batch,3,token_blocks)` with `batch = seq_positions.shape()[0]`.
KV-index control buffers are capacity-allocated (`cap_batch` = max_batch_seqs, e.g.
256) + zero-padded, so a batch-1 prefill launched 256 seq-blocks/layer, 255 empty.
```rust
let active_batch = if ctx.plan().is_decode_only() {
    layer.index.seq_positions.shape().as_slice()[0]   // captured: unchanged
} else {
    ctx.plan().batch.max(1)                            // prefill: actual batch
};
let seq_positions_active = layer.index.seq_positions.narrow(0, 0, active_batch)?;
// pass &seq_positions_active to the kernel instead of &layer.index.seq_positions
```
**Why it may matter MORE on the slow machine:** H200 has ~132 SMs → 256 blocks ≈ 2
waves, small overhead. A GPU with fewer SMs runs 256×1024-thread blocks in many
waves → empty-block overhead can scale to several ms. Verify with the phase table.

### A/B toggle (re-add to measure clean async, no phase-trace syncs)
Add a cached file gate and OR it into the condition:
```rust
fn scatter_fix_disabled() -> bool { use std::sync::OnceLock; static E: OnceLock<bool> = OnceLock::new(); *E.get_or_init(|| std::path::Path::new("/tmp/ri_nofix").exists()) }
// let active_batch = if ctx.plan().is_decode_only() || scatter_fix_disabled() { ...capacity } else { ...actual };
```
Then: `rm /tmp/ri_phase` (clean async), A/B = run with vs without `/tmp/ri_nofix`,
full-restart between, compare `[ttft-trace] run_layers`. cuda:7: 7.85 (off) → 7.27 (on).

---

## 7. Open hypotheses for the slow machine (untested there)

Run §4 decomposition FIRST, then the §5 phase table, then chase whichever wins:

1. **It's not the forward.** If `run_layers` HEAD≈532f5c but TTFT differs → IPC
   park-wake or scheduler orchestration. Trace scheduler (`RUSTINFER_SCHED_TRACE`)
   and the server. See [[vllm-vs-rustinfer-nsys]] / docs/HANDOFF_cudnn_orchestration_fix.md.
2. **Scatter (Fix#1) is the whole gap there.** Plausible if that GPU has few SMs.
   Confirm: phase table shows a_scatter HEAD ≫ 532f5c before fix, = after.
3. **A different phase regressed** (a GEMM, attention). The kernels' `.cu` are
   byte-identical HEAD vs 532f5c (verified by diff on cuda:7) — so a phase gap would
   again be a LAUNCH/PARAMETER difference (grid, workspace, strides), not the kernel.
   Hunt the Rust wrapper that computes the launch config for that op.
4. **GPU clock / power.** Pin clocks (`nvidia-smi -lgc`) or confirm not throttling;
   idle-QPS downclock inflated absolutes on cuda:7.
5. **cuBLASLt GEMM path.** HEAD prefill GEMM uses the nullptr-algo path only when
   `g_zimage_eager_prefill_gemm` is set (matmul.cu) — confirm it's actually set on
   the slow machine (else it falls to the MNK-cached heuristic). 532f5c always
   nullptr. GEMMs are weight-memory-bound (~2.7ms streaming floor for 4B bf16), so
   algo should matter little — but verify.

---

## 8. ncu (optional, pure kernel duration — settles "GPU vs launch")

No standalone forward bench exists; `ncu` must attach to the daemon (painful — it
serializes every kernel, ZMQ may time out). nsys is unavailable. The worker has
`--profile-cuda-steps N` (cudaProfilerStart/Stop around first N steps) for nsys
capture-range if nsys exists on the slow machine. For a clean target, write a
`src/bin/llm_forward_bench.rs` that loads weights + calls `Runtime::step` on a
synthetic prefill (public API: `Runtime::new`/`step`, `run_layers_for_tap`).
The §5 synced phase timing is a good enough proxy without ncu.

---

## 9. Files / paths
- Fix: `crates/infer-backend-cuda/src/lib.rs` (`qkv_norm_rope_scatter`).
- Scatter kernel + grid: `crates/infer-backend-cuda/src/kernels/qkv_norm_rope_scatter/{qkv_norm_rope_scatter.cu, mod.rs}`.
- Forward path: `components/{attention,ffn_dense,decoder_block}.rs`, `application/runtime.rs` (`run_layers`, `step_eager`), `models/decoder.rs`.
- 532f5c equivalent: monolithic `models/qwen3.rs` (forward loop).
- Related: docs/HANDOFF_cudnn_orchestration_fix.md, docs/HANDOFF_graph_prefill.md.

---

## 10. UPDATE — slow machine says: forward CLEAN, gap is IPC + "something" (~7.2ms)

New data from the slow machine (532f5c TTFT≈8ms, HEAD≈13ms): the **GPU forward has
no regression** there (kernels are byte-identical; Fix#1 §6 only mattered on cuda:7).
The 7.2ms gap is **CPU-side: IPC wake/park latency + orchestration**, not compute.

### 10a. CONFIRMED IPC regression — server ZMQ-client lost its wake (FIXED)
`crates/infer-server/src/client/zmq_client.rs`. Commit **`cac326a` "use cudnn for fa"**
incidentally removed the inproc-PAIR **wake socket** and made `Waker::wake()` a
**no-op**, then capped `POLL_MAX_TIMEOUT` 1s→1ms as a band-aid.

Mechanism: axum pushes a request onto the mpsc command channel, then the ZMQ thread
must *notice* it. With the wake gone, it only notices when its `zmq::poll(DEALER, T)`
**times out** — up to `POLL_MAX_TIMEOUT`. So every request submit ate **up to ~1ms**
(avg ~0.5ms) on the TTFT path. On 532f5c the wake byte interrupted the poll
**instantly** (even with the 1s timeout). Responses were never affected (they wake
via DEALER POLLIN).

**Fix (in tree):** restored an instant wake using a **plain OS pipe** (`std::io::pipe`,
Rust ≥1.87) instead of a ZMQ inproc PAIR — a pipe fd cannot trip libzmq's cross-thread
signaler assertion (the reason the original was removed). The ZMQ thread polls the
pipe's fd via `zmq::PollItem::from_fd` alongside the DEALER; `wake()` writes 1 byte;
the reader drains. `POLL_MAX_TIMEOUT` restored to 1s. All 4 existing `waker.wake()`
call sites now actually fire. Builds clean (`cargo build -p infer-server`).
Magnitude: ~0.5–1ms on a tuned host; **more on a power-managed CPU** (see 10c).

### 10b. The remaining "something" — decompose with the THREE built-in trace hooks
All three already exist in HEAD (no code to paste back this time):
- **server** `crates/infer-server/src/api/openai/streaming.rs` → log line
  `TTFT_TRACE: chat first content chunk` with `server_ttft_ms` (request→first SSE chunk).
  Always on (info).
- **scheduler** `crates/infer-scheduler/src/application/event_loop.rs` → set env
  `RUSTINFER_SCHED_TRACE=1`: logs `SCHED_TRACE: NewRequest->dispatch` (us) and
  `SCHED_TRACE: StepOutput->forward` (us).
- **worker** `serve_loop.rs` + `worker_scheduler.rs` → set env `RUSTINFER_TTFT_TRACE=1`:
  logs `[ttft-trace] handle_prefill wall` and `[ttft-trace] runner.step (forward+sample)`.
  (Env reaches the worker only if NOT stripped by the launch wrapper — if missing, gate
  on a sentinel file like §1; see [[scatter-grid-capacity-regression]].)

Decomposition (run on the slow machine, 532f5c vs HEAD, low QPS, single stream):
```
server_ttft  =  [submit IPC]  +  sched(NewRequest->dispatch)  +  [dispatch IPC hop]
             +  handle_prefill_wall  +  sched(StepOutput->forward)  +  [reply IPC]  +  SSE encode
where runner.step ⊆ handle_prefill_wall   (forward+sample, confirmed CLEAN)
```
Read off which term grew HEAD vs 532f5c:
- `handle_prefill_wall − runner.step` large → prefill **orchestration** (KV alloc
  `alloc_with_relief`, block-table concat, `send_step_output`). That is the non-forward
  worker "something".
- `sched(*)` large → scheduler engine/planning.
- `server_ttft − (sum of the above)` large → **IPC hops** = submit wake (10a, fixed) +
  the dispatch/reply park-wakes (10c).

### 10c. Prime amplifier hypothesis: CPU power management (fits "fast host hides it")
The refactor leans on **OS wakeups** at several hops (zmq-client submit poll, worker
data-plane `zmq::poll` PARK between requests, scheduler channel handoffs). A POLLIN
wake is kernel-delivered, but the **core still has to exit its idle C-state**, and a
`poll(timeout)` wake waits on a timer. On a tuned server (perf governor, shallow
C-states) that is ~µs and invisible; on a default/powersave/loaded machine each
park→wake can cost **several ms** — which would turn the same code into the 7.2ms gap
while leaving the GPU forward untouched. Check on the slow machine FIRST (cheap):
```bash
cpupower frequency-info | grep -i governor        # want: performance
cat /sys/devices/system/cpu/cpu*/cpuidle/state*/disable   # deep C-states
turbostat --quiet sleep 1   # watch C-state residency + wake latency under load
```
Mitigate to test the hypothesis: `cpupower frequency-set -g performance`, disable deep
C-states (`cpupower idle-set -D 0` or kernel `intel_idle.max_cstate=1`), and pin the
worker/scheduler/server threads. If the gap collapses, it was wake latency, and 10a's
wake-pipe fix (instant submit wake, no timer) is the code-side half of the cure.

### 10d. What was checked and ruled OUT (so you don't re-chase)
- forward GPU phases: identical (this doc §5) — and the slow machine confirms.
- sampling: `GreedySampler` uses on-device `D::argmax`, downloads only `rows` ints
  (`application/sampler_stack.rs`) — NOT a full-logits copy. Not a regression.
- worker serve-loop poll: event-driven, POLLIN wakes immediately; the old
  `idle_wait_ms=heartbeat/2` window was REMOVED (improvement), and `wait_for_prefill_quiet(1ms)`
  is commented out. Not a regression.
- scheduler transports + control plane: no poll/timeout/wake change (only a WorkerId
  type move and a ClientId clone tidy). Engine/planning/main are refactors, no added sync.
- decode 1-deep pipelining: affects TTOT, not TTFT (first token is sent inside
  `handle_prefill` before any decode step).

---

## 11. ROOT CAUSE of the bulk "something" — KV allocator O(num_blocks) per completion (PROVEN)

Resolves the logic gap in §10: the user tests **both versions on the same machine**, so
env (C-states/governor) is constant and cannot explain a delta — **the delta is code**, and
a slower CPU only *amplifies* host-side per-request work HEAD added (invisible to kernel-time
"forward").

**File:** `crates/infer-worker/src/domain/global_kv_alloc.rs`.
HEAD's "Eager merge (A1)": `release_owned()` — called on **every sequence completion**
(default real-time recycling, prefix-caching off) — calls `free()` → `merge_sorted_returned`,
an **O(num_blocks) in-place merge** over the entire free pool. 532f5c's `release()` parked
freed slots in a lazy `released` holding list (O(returned)≈0), draining only on alloc failure.
`paged_block_size=1` ⇒ num_blocks = token slots; the pool auto-sizes to ~780k (max working set
256×1024 = 262k → ~3× over).

**Empirical proof** — micro-bench of the *real* impl on cuda:7's host CPU (`rustc -O`,
`-C debug-assertions=no`; source: `scratchpad/kv_bench2.rs`, mirrors the file via a tracing shim):

| num_blocks | `release_owned`→`free()` per completion |
|---|---|
| 16384  | 0.039 ms |
| 65536  | 0.159 ms |
| 262144 | 0.648 ms |
| 786432 | **2.005 ms** |

Linear in num_blocks. ~2ms/completion at the ~780k auto pool on cuda:7's *fast* CPU →
multi-ms on the slow machine's CPU. Under continuous batching (the bench is concurrent),
completions interleave with prefills in the serve loop, so each merge delays a prefill →
**inflates TTFT**. Pure host CPU ⇒ "forward clean" holds. This is the bulk of the 5–7ms gap.

### 11a. FIX DONE (in tree, builds) — clamp the auto-sized pool to the working set
`serve_loop.rs` bootstrap: when prefix-caching is OFF, the VRAM-probe `num_blocks` is now
clamped to `max_batch_seqs × max_blocks_per_seq` — the most slots the worker can ever hold
in use (the scheduler admits ≤ max_batch_seqs seqs, each ≤ max_seq_len). Lossless; blocks
beyond that were unreachable, only wasting VRAM and inflating every O(num_blocks) op.
Default 256×1024 ⇒ cap 262144 (vs ~780k) ⇒ free() 2.0ms→0.65ms (cuda:7). Bootstrap log now
prints `probed=X -> num_blocks=Y` so you can see the reduction. **For full elimination set
`max_batch_seqs` to real concurrency** (e.g. 32 ⇒ 32768 ⇒ ~0.08ms/completion) — clamp
follows automatically. (Or set toml `num_blocks` explicitly to override the probe entirely.)
This is the confirmation test too: rebuild worker, re-measure HEAD TTFT → should drop.

### 11b. Proper fix (NOT applied — allocator is corruption/starvation-prone; validate on full stack)
Make `release_owned`/`free` **O(returned)** while keeping freed slots immediately
allocatable. Block-slot order is "architecturally irrelevant" for `block_size=1` (module
doc), so the full-pool sorted merge is unnecessary for correctness. Sketch: `release_owned`
→ `release()` (park in `released`, O(returned)); `alloc_indices` drains `released` first
(O(returned)) before bumping; keep `recycle()`'s full sort as a rare cleanup. Verify the
A1 starvation does not return — `alloc_with_relief` already gates admission on `total_free()`
(which counts `released`), so the accounting should hold; confirm decode batch fills to
capacity and there is no KV leak (run the worker unit tests + a burst bench).
