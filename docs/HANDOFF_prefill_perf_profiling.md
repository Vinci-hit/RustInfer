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
