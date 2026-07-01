# RustInfer Code Review — 2026-07

> Scope: entire backend (all crates **except** `infer-frontend`). Generated from a multi-agent review —
> 17 code slices each read in full by a dedicated reader, then every falsifiable *high/critical* claim was
> re-checked by an independent adversarial verifier. **24 findings were refuted** on verification and are
> listed separately at the bottom so they are not chased again.
>
> **Diffusion issues are recorded but deferred** (marked `DEFERRED — diffusion`) per current priorities.
>
> **Fix pass (non-diffusion): 21 fixed ☑, 43 won't-fix ⊘** (each with rationale in the tracker). All fixes
> compile (`cargo check --workspace --exclude infer-frontend`) and the touched crates' test suites pass
> (`infer-core`, `infer-backend-cpu`, `infer-scheduler` = 144 tests; `infer-worker` lib = 104 tests, run
> single-threaded — the parallel run SIGSEGVs on a pre-existing single-CUDA-context test-harness issue).

## Summary

| Effective severity | Total | Non-diffusion | Diffusion (deferred) |
|---|---|---|---|
| 🔴 critical | 2 | 0 | 2 |
| 🟠 high | 12 | 10 | 2 |
| 🟡 medium | 60 | 55 | 5 |
| ⚪ low | 33 | 29 | 4 |
| **Active total** | **107** | **94** | **13** |
| ⚫ refuted (excluded) | 24 | | |

Severity reflects the **verified** severity where a verifier adjusted it (original severity noted inline).

> **F131** (🟠 high, ☑ fixed) was found *after* the original multi-agent sweep, during an end-to-end
> smoke test of Qwen3-4B — a bootstrap KV-sizing OOM. It is included in the counts above and detailed
> in *Worker · Serve loop*.

## Fix tracker (non-diffusion)

Status: ☑ fixed · ⊘ won't-fix/by-design (rationale in the Notes column) · ☐ open

| ID | Sev | Status | File:Lines | Title | Notes |
|---|---|---|---|---|---|
| F074 | 🟠 high | ☑ | `crates/infer-core/src/tensor.rs:237` | Narrow may mark non-contiguous view as contiguous | `narrow` now keeps `dim==0` slices contiguous (a dim-0 sub-block stays contiguous). |
| F073 | 🟠 high | ☑ | `crates/infer-core/src/tensor.rs:281-286` | data_ptr arithmetic may overflow on large offsets | Added a debug-assert bounds check in `view_raw` (offset+numel ≤ storage bytes); validates once at view construction, hot-path `data_ptr` stays unchecked. |
| F108 | 🟠 high | ☑ | `crates/infer-backend-cpu/src/lib.rs:181-194` | Integer overflow in ewise_mul lacks input bounds validation | `ewise_mul` now calls `check_contiguous3`/`check_numel3` like the other elementwise ops. |
| F110 | 🟠 high | ☑ | `crates/infer-backend-cpu/src/lib.rs:290-309` | Potential out-of-bounds access in embedding operation | `embedding` now validates every index against `[0, vocab)` before the raw copy. |
| F109 | 🟠 high | ☑ | `crates/infer-backend-cpu/src/lib.rs:357-385` | Unsafe pointer arithmetic without bounds checks in split_cols | `split_cols` now checks contiguity and that `rows*total_cols`/`rows*dst_cols` fit the backing tensors. |
| F070 | 🟠 high | ☑ | `crates/infer-scheduler/src/infrastructure/transport/zmq_transport.rs:44-45, 234-235` | ZMQ unbounded channels risk OOM under load | Both scheduler→ZMQ outbound queues (frontend responses, worker commands) are now bounded (16384) with async `send().await` backpressure; inbound stays unbounded. |
| F021 | 🟠 high | ☑ | `crates/infer-worker/src/models/loader.rs:193` | Same silent truncation issue in fused QKV loading | Loader now errors on a safetensors byte-length mismatch instead of silently zero-padding (fixed at 3 sites: fused_qkv, fused_gate_up, single-tensor loader). |
| F131 | 🟠 high | ☑ | `crates/infer-worker/src/application/serve_loop.rs:78-201` | KV-cache sizing OOMs at bootstrap: free-memory probe taken *before* the activation workspace is allocated | Now profile-driven: build Runtime with a tiny throwaway KV pool → run one worst-case eager dummy forward (commits the GiB-scale logits workspace + lazy cuBLASLt/cuDNN allocs) → probe real free memory → size and reallocate the real KV pool → then capture graphs. Verified end-to-end at `mem_fraction_static=0.9` (the value that OOM'd before). New `Runtime::profile_forward`/`resize_kv_pool`; new `PROFILE_KV_BLOCKS`/`PREWARM_HEADROOM_BYTES` tuning constants. |
| F039 | 🟠 high | ☑ | `crates/infer-worker/src/bin/worker_main.rs:249` | Config parsing lacks device existence validation in worker_main | Real device string is now threaded from `cfg.device` into `ControlPump`; Hello/Ready report the actual device (also covers F040). |
| F000 | 🟠 high | ⊘ | `crates/infer-worker/src/application/runtime.rs:2654-2672` | Use-after-free in upload_i32_full_zeropad when async H2D DMA completes after function returns | Not a live UAF: `cudaMemcpyAsync` from **pageable** host memory stages synchronously before returning (the codebase's own `block_tables_host` comment confirms this), so the local Vec is safe to drop. The buffer is ~1 KB (vs the 1–4 MB block table that WAS worth fixing). A persistent staging buffer would help only if the host side is ever pinned; revisit then. |
| F086 | 🟡 medium | ⊘ | `crates/infer-core/src/dtype/mod.rs:26-27` | DTypeId::register uses relaxed atomics without synchronization | Relaxed monotonic counter is correct for unique-id allocation; the RwLock write establishes happens-before for the spec insert. Startup-only path. |
| F081 | 🟡 medium | ⊘ | `crates/infer-core/src/dtype/mod.rs:28-31, 40-46, 53-59` | Dtype registry uses RwLock with poisoning but no recovery | Registry writes are startup-only (dtype registration); poisoning on a panicking registration is acceptable fail-fast, not a runtime DoS vector. |
| F088 | 🟡 medium | ⊘ | `crates/infer-core/src/exec.rs:130-136` | Workspace pointer is not validated for alignment or size | `Workspace::from_raw` is an internal constructor fed by the backend's own arena; callers pass correctly-sized/aligned pointers by construction. |
| F082 | 🟡 medium | ⊘ | `crates/infer-core/src/kv.rs:167-172` | KvEdit::apply_step may incorrectly compute truncate range | False alarm: `accepted<=spec` and `current>=spec` are both checked, so `keep=current-spec+accepted<=current` (no overflow); `&mut self` means no concurrency. |
| F084 | 🟡 medium | ⊘ | `crates/infer-core/src/ports/fused_ops.rs:391` | scatter_kv_paged_reference may read out-of-bounds if block_tables is corrupted | Reference/CPU path only; the bounds check at fused_ops.rs:391 does gate the access. Prod runs the CUDA kernel. |
| F075 | 🟡 medium | ⊘ | `crates/infer-core/src/storage.rs:38-44` | Double-free risk on storage due to missing size validation | `size.max(1)` is symmetric between alloc and free (Drop also uses `size.max(1)`), and the CUDA pool keys by the requested size — no asymmetry or double-free. |
| F078 | 🟡 medium | ⊘ | `crates/infer-core/src/tensor.rs:160-164` | to_host_vec may succeed with empty tensor but violates Vec invariants | `Vec::with_capacity(0)` + `set_len(0)` is well-defined (empty Vec, dangling-but-aligned ptr); no allocator-contract violation. Verifier-adjacent to the refuted claims. |
| F077 | 🟡 medium | ⊘ | `crates/infer-core/src/tensor.rs:295-298` | Unsafe assume about storage alignment in data_ptr_mut without validation | Backends allocate with alignment ≥ the dtype's requirement by construction; `data_ptr_mut` mirrors `data_ptr`. Adding per-call alignment checks on the hot path isn't warranted. |
| F080 | 🟡 medium | ⊘ | `crates/infer-core/src/tensor.rs:366` | copy_from may copy overlapping memory even after checking storage pointers | `copy_from` compares `Arc::as_ptr` for the common aliasing case; cross-device overlap of distinct allocations isn't a real scenario in this engine (one device per worker). |
| F112 | 🟡 medium | ☑ | `crates/infer-backend-cpu/src/lib.rs:219-246` | Missing contiguity checks in matmul_quant | `matmul_quant` now checks contiguity, `group_size` divisibility, and operand sizes vs declared m/n/k. |
| F113 | 🟡 medium | ⊘ | `crates/infer-backend-cpu/src/lib.rs:873-882` | Unsafe cast from *const T to *const u8 loses alignment information in read_f64 | Sound: casting to `*const u8` gives alignment-1, and reading bytes has no alignment requirement regardless of T. No UB. |
| F094 | 🟡 medium | ☑ | `crates/infer-backend-cuda/src/config.rs:243-246` | Arena overflow rollback is not atomic (TOCTOU race) | Arena `arena_alloc` now reserves via `fetch_update` (never transiently over-commits) instead of fetch_add+rollback. |
| F092 | 🟡 medium | ⊘ | `crates/infer-backend-cuda/src/kernels/matmul/sdpa.rs:187-198, 259, 296` | Per-token allocation churn in eager SDPA (non-graph-captured decode) | Only the non-graph-captured eager fallback allocates; the primary decode path uses the capture arena (already zero-alloc). Eager is a correctness fallback, not the hot path. |
| F093 | 🟡 medium | ☑ | `crates/infer-backend-cuda/src/lib.rs:706-715, 513-527` | Missing workspace size validation in release builds | Flash-decode workspace size check is now ALWAYS on (returns OpError) instead of `#[cfg(debug_assertions)]` — prevents silent OOB device writes in release. |
| F100 | 🟡 medium | ☑ | `crates/infer-backend-cuda/src/kernels/broadcast_mul/mod.rs:65, 110` | Unchecked division in broadcast kernels | `broadcast_mul_inplace`/`broadcast_add_inplace` now validate `x.numel() % dim == 0` and `dim != 0` before the division. |
| F105 | 🟡 medium | ⊘ | `crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:106, 172` | Potential integer overflow in byte size calculations in pad.rs | Byte-size products are bounded by tensor capacity (allocated from VRAM, far below usize::MAX). Cannot overflow in practice. |
| F098 | 🟡 medium | ⊘ | `crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:122, 129` | Unchecked usize to i32 casting can cause integer overflow | Casts are bounded by model dims (hidden ≤ ~16k, tokens ≤ max_batch_tokens=8192), far below i32::MAX; cannot overflow with real weights. Guarding dozens of launch wrappers is churn. |
| F103 | 🟡 medium | ⊘ | `crates/infer-backend-cuda/src/kernels/rope/mod.rs:60-71` | No validation that device positions pointer is valid in rope kernels | `positions_dev` is produced by the worker's own upload path with a known length; it's an internal contract, not external input. |
| F106 | 🟡 medium | ⊘ | `crates/infer-backend-cuda/src/kernels/swiglu/mod.rs:110-111` | Per-token memory allocation in swiglu_packed F32 fallback | Only the F32 fallback allocates; production runs BF16. Not the hot path. |
| F128 | 🟡 medium | ☑ | `crates/infer-protocol/src/config.rs:279-312` | Potential dispatch inconsistency: resolve_model_type silent fallback to 'llama3' | `resolve_model_type` now warns on empty/unrecognized architecture hints before defaulting to llama3. |
| F129 | 🟡 medium | ⊘ | `crates/infer-protocol/src/worker_to_scheduler_data.rs:20-28` | No validation that token_ids in AssignedIndices serialized size stays within bounds | token_ids are only populated when prefix caching is enabled (opt-in); when on, they're the intended RadixTree payload. `is_consistent()` (F127) now guards malformed runs. |
| F127 | 🟡 medium | ☑ | `crates/infer-protocol/src/worker_to_scheduler_data.rs:39-41` | Lack of overflow validation for token_ids.len() vs assigned_indices.len field | Added `AssignedIndices::is_consistent()` protocol helper; the scheduler consumer already warns+skips on mismatch (output_fns.rs:211). |
| F124 | 🟡 medium | ☑ | `crates/infer-protocol/src/worker_to_scheduler_data.rs:45-47` | Integer overflow in AssignedIndices.end() with unconstrained indices | `AssignedIndices::end()` uses `saturating_add`; added `is_consistent()` to check overflow + token_ids/len match. |
| F125 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/application/output_fns.rs:265-266` | Unnecessary String clones on hot path (per-token streaming) | The per-token String is required by the `StreamChunk.request_id: String` wire type; an Arc<str> would still materialize a String at the wire boundary. Cost is dwarfed by per-token msgpack+ZMQ send. |
| F126 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/application/output_fns.rs:338, 351-352, 357` | Redundant clones of external_id and image data in diffusion output path | Diffusion output path — deferred per current priorities (diffusion out of scope). |
| F058 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/application/batch_builder.rs:175` | Per-segment prefix hint clone in hot path | Per-prefill-segment clone (not per-token), bounded by batch size; `prefix_hints` is a shared borrow so it can't be moved out. The code comment already documents the bounded linear scan. |
| F061 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/application/workflow/llm.rs:276-344` | Per-step allocation in sanitize_step_output on stale sequences | Clone only triggers on the stale-sequence slow path (cancelled/finished), which is rare; the common clean path is untouched. Optimizing the rare path adds complexity for little gain. |
| F065 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/domain/policy/continuous_batching.rs:256-257` | Incorrect decode reserve allows sequence budget bypass | False alarm: `max_seqs` is enforced separately (`seqs_used >= seq_budget`). `decode_reserve` only governs KV budget, and a max_tokens=1 request genuinely needs 0 decode slots (its token comes from prefill). |
| F071 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/infrastructure/transport/control_plane/pending_calls.rs:205-216` | Potential integer overflow in RequestId allocator on skip-zero | Requires 2^64 requests to hit; the skip-zero double-increment is correct for any realistic run. Not worth guarding. |
| F054 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:198-207, 458` | Memory leak risk: free_ids reuse does not check for stale LRU generation collisions | Documented design: reused ids keep a stale (higher) generation stamp so lingering queue entries are correctly skipped; `compact()` bounds the map. Verifier found no unbounded growth in practice. |
| F053 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:438-465` | LRU eviction does not promote parent after child removal | On close reading the code already calls `maybe_promote_to_lru(parent)` after detaching a child (radix_tree.rs:459-461); the reviewer's own note concludes it 'is actually handled'. No bug. |
| F047 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/domain/inference_session/queue.rs:127-131` | Lazy removal tombstone compaction threshold heuristic | Heuristic threshold; verified correct. Pathological tombstone accumulation is a theoretical latency spike, not incorrectness. Tuning left for a profiling-driven change. |
| F044 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/domain/inference_session/table.rs:222-224` | Per-iteration Vec allocation on hot path in prefilling() | Bounded (≤ max_batch_seqs pointers), once per scheduling iteration, dwarfed by the per-step msgpack+network I/O. Converting to an iterator would invade the batch-builder slice API for negligible gain. |
| F045 | 🟡 medium | ⊘ | `crates/infer-scheduler/src/domain/inference_session/table.rs:313-322` | Per-iteration Vec allocation in prefilling_continuations() | Same as F044: small bounded Vec once per iteration; `has_prefilling_continuations()` already short-circuits the empty case with no allocation. |
| F046 | 🟡 medium | ☑ | `crates/infer-scheduler/src/domain/inference_session/table/accounting.rs:34-38` | Missing validation in decoding_kv_slots integer cast | `decoding_kv_slots` feeds `preemption_candidates`, which now clamps all usize→u32 casts via `u32::try_from().unwrap_or(MAX)`. |
| F049 | 🟡 medium | ☑ | `crates/infer-scheduler/src/domain/inference_session/table/accounting.rs:98-110` | Integer cast from usize to u32 in PreemptCandidate without overflow check | `preemption_candidates` now clamps output_len/input_len/kv_used with saturating `u32::try_from`. |
| F119 | 🟡 medium | ⊘ | `crates/infer-server/src/api/openai/images.rs:121` | No timeout on parallel image generation futures | Already bounded: `infer()` registers a per-request deadline enforced by the ZMQ thread's `cancel_timed_out_requests`, so a hung image request is capped at `request_timeout_secs`. |
| F117 | 🟡 medium | ☑ | `crates/infer-server/src/api/openai/streaming.rs:19-30` | Per-token JSON serialization failure swallows error detail | SSE serialization failure now emits a distinguishable `error` event instead of a fake `[DONE]` terminator. |
| F122 | 🟡 medium | ⊘ | `crates/infer-server/src/client/zmq_client.rs:108-115` | Mutex contention on Waker pipe per command submit | Mutex is held only for a 1-byte pipe write, never across await; effectively uncontended. A prior inproc-PAIR waker was removed deliberately (see comment). Not a real bottleneck. |
| F118 | 🟡 medium | ⊘ | `crates/infer-server/src/client/zmq_client.rs:23, 536` | Stream buffer exhaustion under per-token backpressure | Buffer size is a tuning constant; the ZMQ client already re-arms the deadline on each chunk (zmq_client.rs:404), so slow-but-alive clients aren't falsely cancelled. |
| F120 | 🟡 medium | ⊘ | `crates/infer-server/src/client/zmq_client.rs:349-436` | Double deserialization per stream chunk in ZMQ handler | Reordering msgpack deserialization by tag is risky (struct disjointness isn't guaranteed) and the cost (~µs) is dwarfed by the ZMQ recv syscall. Not worth the correctness risk. |
| F121 | 🟡 medium | ⊘ | `crates/infer-server/src/client/zmq_client.rs:450-459` | Non-atomic deadline check before sending timeout chunk | Single-threaded ZMQ client thread owns `pending`; the 'TOCTOU' is within one thread's sequential code, so no actual race. |
| F042 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/serve_loop.rs:730` | Integer cast overflow risk in heartbeat KV stats | Already correct: line 730 uses `.min(u32::MAX as usize) as u32` — a saturating clamp, not a silent truncation. |
| F038 | 🟡 medium | ☑ | `crates/infer-worker/src/infrastructure/io/safetensors.rs:168` | Fallback scan of shard[0] assumes non-empty shards vector | `read_view`/`contains` now return an error instead of panicking on empty shards or an out-of-range shard index. |
| F040 | 🟡 medium | ☑ | `crates/infer-worker/src/infrastructure/transport/control_pump.rs:55, 115` | Hardcoded CUDA device assumption in control messages | `ControlPump` now carries the real device string; Hello/Ready no longer hardcode `cuda:0`. |
| F014 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/kv_relief.rs:204-209` | Convoluted shrink_to_active retry loop in alloc_with_relief may be hard to reason about | Correctness confirmed; this is a readability/maintainability note. Left as-is to avoid behavioral risk in the relief path. |
| F012 | 🟡 medium | ⊘ | `crates/infer-worker/src/domain/forward_scratch.rs:157-167` | Unsafe data race in flash_workspace_mut due to undocumented GPU parallelism | Documented single-stream invariant: all layers run serially on one CUDA stream, so the `UnsafeCell` view is never aliased. This is the same pattern used throughout; no live race. |
| F013 | 🟡 medium | ⊘ | `crates/infer-worker/src/domain/global_kv_alloc.rs:99-105` | Potential panic in total_free due to unchecked head >= len invariant | `head <= free.len()` is an internal invariant upheld by every mutator; the fuzz test `fuzz_no_duplicates_no_loss_no_panic` exercises this. A debug_assert could document it but it's not a live bug. |
| F003 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/runtime.rs:1876-1897` | D2H copy_out_mixed_abc may transfer stale data from padded forward results | Verifier + inspection: D2H copies exactly `batch` elements which the merge kernels wrote; `finalize_*` bounds-check against active_n+finished_n. Copy-out stream is drained between steps. No stale read in practice. |
| F001 | 🟡 medium | ☑ | `crates/infer-worker/src/application/runtime.rs:1933-1935` | Integer overflow in token_offset accumulation corrupts last_token_rows indices | `token_offset` accumulation now uses `checked_add` and returns a Shape error on overflow instead of wrapping i32. |
| F004 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/runtime.rs:2226-2230` | Subtle device control buffer state machine with reuse_device_control flag | Documented, subtle-but-correct: steps synchronize the copy-out stream (`finalize_decode_abc`) before reinterpreting control buffers; `reuse_device_control` only skips re-upload when the prior batch is compatible. |
| F008 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/decode_engine.rs:625-635` | Speculative KV slot reservation failure not tested for reclaimed prefill token integrity | The failure path is handled gracefully (check at decode_engine.rs:689); this is a design-fragility note, not a live defect. |
| F009 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/worker_scheduler.rs:363-369` | Decode rows not re-synced before critical build_decode_request after control drain in fused step | Verifier refuted the sibling claim (F-decode 854-858): control drain happens before `decode_order` is materialized in the fused path, and the .expect() is guarded. No live panic. |
| F007 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/worker_scheduler.rs:510-632` | Pad row sentinel u64::MAX not filtered from ABC next-rows recording | Safe by construction: `decode_order` never contains pad rows, so u64::MAX sentinels can't leak into `abc_next_rows`. This is a 'if the design changes' caution, not a live bug. |
| F011 | 🟡 medium | ⊘ | `crates/infer-worker/src/application/worker_scheduler.rs:645-671` | Missing assertion on decode_order length consistency across fused prefill groups | Design/assertion suggestion only; the cursor invariant holds today. Could add a debug_assert later, but not a correctness bug. |

Low-severity non-diffusion items are listed in their slice sections below (not tracked individually).

---

## Findings by area

### Worker · Runtime (forward / CUDA graph / KV plan)

> The worker runtime is the core hot path for batch planning, KV management, CUDA graph capture/replay, and mixed decode+prefill forwarding. The code is architecturally sophisticated with persistent GPU buffers for graph stability, async D2H pipeline for overlapped compute, and careful buffer lifecycle management. However, several critical soundness issues exist: (1) a use-after-free in upload_i32_full_zeropad where a local Vec is dropped before async H2D completes, (2) integer overflow in token_offset accumulation that corrupts last_token_rows indices, and (3) potential D2H copy buffer mismatches when batch size exceeds or changes between steps. The design is generally sound for the async decode pipeline, but FFI/lifetime boundaries lack proper verification.

#### F000 · 🟠 high · [soundness] Use-after-free in upload_i32_full_zeropad when async H2D DMA completes after function returns · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/runtime.rs:2654-2672`
- **Verified:** confirmed (severity critical→high)
- **Resolution:** ⊘ Not a live UAF: `cudaMemcpyAsync` from **pageable** host memory stages synchronously before returning (the codebase's own `block_tables_host` comment confirms this), so the local Vec is safe to drop. The buffer is ~1 KB (vs the 1–4 MB block table that WAS worth fixing). A persistent staging buffer would help only if the host side is ever pinned; revisit then.

**Problem.** In upload_i32_full_zeropad, a local Vec<i32> `padded` is allocated, filled, and then upload_async is called with padded.as_ptr(). The function returns immediately after upload_async (no synchronize), so the Vec's host memory may be freed before the H2D DMA completes. The device.upload_async documentation says 'src must remain valid until the device stream consumes the copy.' Since upload_i32_full_zeropad returns immediately, the caller may deallocate or reuse the stack frame, causing the DMA to read from freed memory. This is a classic use-after-free: the GPU reads a pointer that is no longer valid on the host.

```rust
Lines 2667-2671: `let mut padded = vec![0i32; cap]; padded[..host.len()].copy_from_slice(host); let bytes = std::mem::size_of_val(padded.as_slice()); ... unsafe { device.upload_async(ptr, padded.as_ptr() as *const u8, bytes) }` The `padded` Vec is local and dropped at function exit, before the async H2D may complete.
```

**Fix.** Use the persistent host staging buffer pattern (block_tables_host, prefill_ids_host) applied elsewhere in the Runtime. Either allocate the padded buffer once in the Runtime struct and reuse it (zero-ing before each upload), or call the synchronous device.upload (which synchronizes internally) instead of upload_async.

> **Verifier note:** Use-after-free in upload_i32_full_zeropad and related upload functions: local Vecs (padded, cu_q_lens, block2req, block2tile) are created, used with upload_async for H2D copies, and immediately deallocated before the GPU stream consumes the data. While unlikely to manifest in practice due to fast GPU execution on same stream and small buffer sizes, this violates the documented safety contract of upload_async and could cause data corruption or crashes under specific timing conditions (delayed GPU execution, rapid function re-entry causing stack reallocation). The correct pattern (persistent buffers that outlive async operations) is already demonstrated elsewhere in the code.

#### F003 · 🟡 medium · [soundness] D2H copy_out_mixed_abc may transfer stale data from padded forward results · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/runtime.rs:1876-1897`
- **Resolution:** ⊘ Verifier + inspection: D2H copies exactly `batch` elements which the merge kernels wrote; `finalize_*` bounds-check against active_n+finished_n. Copy-out stream is drained between steps. No stale read in practice.

**Problem.** In copy_out_mixed_abc, the D2H copies read `batch * elem` bytes from device buffers (argmax_out_dev, active_src_rows_dev, etc.) into host mirrors. The device buffers were written by the forward+merge kernels. However, if the forward was padded to a larger slot_batch (e.g., via ceil_capture_slot for graph replay), the device buffers may contain uninitialized data in rows [batch, slot_batch). The D2H copies exactly batch elements, which is correct IF the kernels only wrote batch valid elements. But if a prior step had larger batch and those buffers retain stale data, the D2H copies stale data. More critically, the finalize_mixed_abc function (line 2041) does index bounds checking on active_src_rows_host and finished_src_rows_host, expecting valid data only up to active_n + finished_n. If the D2H copies stale/garbage data, bounds checks may pass spuriously.

```rust
Lines 1876-1897: `let row_bytes = batch * elem;` then D2H copies of argmax_out, active_src_rows, finished_src_rows. The host buffers are capacity cap_batch, and D2H reads batch elements, which is safe for buffer overflow. But the device source may contain garbage if the forward was padded.
```

**Fix.** The merge kernel should zero the result buffers or only write valid elements. Alternatively, ensure the forward at padded batch size still produces valid outputs in tail rows (most kernels output per-row independent results, so this is likely safe, but undocumented). Add an assert or validation in finalize_mixed_abc to check that indices are within reasonable bounds.

#### F001 · 🟡 medium · [correctness] Integer overflow in token_offset accumulation corrupts last_token_rows indices · ☑ FIXED

- **Location:** `crates/infer-worker/src/application/runtime.rs:1933-1935`
- **Verified:** confirmed (severity high→medium)
- **Resolution:** ☑ `token_offset` accumulation now uses `checked_add` and returns a Shape error on overflow instead of wrapping i32.

**Problem.** In upload_mixed_abc_metadata, the token_offset is accumulated with unchecked addition of per-row q_lens. If the sum of q_lens across all rows exceeds i32::MAX, the addition wraps and produces incorrect last_token_rows values. This causes the mixed merge kernel to index the wrong rows when selecting which token to output. The merge uses last_token_rows_dev to gather the final token per sequence from the flat token tape. With wrapped indices, wrong tokens are sampled and returned to the user.

```rust
Lines 1933-1935: `let q = req.seqs[i].input_ids.len().max(1) as i32; token_offset += q; self.abc.last_token_rows_host[i] = token_offset - 1;` The `+=` operator on i32 wraps silently on overflow without error. Build_plan validates num_tokens as usize fits in cap_num_tokens, but does not check the i32 cast.
```

**Fix.** Use checked arithmetic: `token_offset = token_offset.checked_add(q).ok_or_else(|| OpError::Shape(format!("token_offset overflow at row {}", i)))?;`. Alternatively, use i64 for token_offset to avoid overflow for realistic batch sizes.

> **Verifier note:** Integer overflow is technically possible in token_offset accumulation if cap_num_tokens is configured above i32::MAX, which would cause incorrect last_token_rows indices in the GPU kernel. However, the practical likelihood is extremely low due to hardware memory limitations and default configuration sizes being orders of magnitude below the overflow threshold. No explicit validation prevents this edge case at configuration time.

#### F004 · 🟡 medium · [correctness] Subtle device control buffer state machine with reuse_device_control flag · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/runtime.rs:2226-2230`
- **Resolution:** ⊘ Documented, subtle-but-correct: steps synchronize the copy-out stream (`finalize_decode_abc`) before reinterpreting control buffers; `reuse_device_control` only skips re-upload when the prior batch is compatible.

**Problem.** The is_decode_abc function assumes that at line 2301-2306, if a graph slot is ready, it will use that slot's batch size (slot_batch >= plan.batch). The code then pads the control buffers to slot_batch by zero-padding in upload_index. However, if a prior step's compact_extend_control call wrote device state for a different batch size, the tail state may corrupt the next step's interpretation. The reuse_device_control flag (line 2226) skips upload_index when true, meaning control buffers are left from the prior step. If that prior step had larger batch, the zero-padded tail from that step remains, which should be safe. However, if prior batch was smaller, the stale data from two steps back could remain. The code does synchronize between steps (finalize_decode_abc drains the copy-out stream), so this should be safe in practice, but the buffer state machine is subtle.

```rust
Lines 2226-2230 and 2301-2306: reuse_device_control skips upload_index, relying on compact_extend_control to have left valid state. But if admission/eviction happens, upload_index is called to re-seed from request. The logic is sound but fragile.
```

**Fix.** The current logic is correct: reuse_device_control only skips upload_index if the prior step's compact_extend already left valid state, and any admission/eviction forces a re-seed. No code change needed, but add a comment documenting the invariant: 'Control buffers are valid iff the prior compact_extend left them (reuse_device_control=true) OR this step is calling upload_index (reuse_device_control=false).' This prevents future refactors from breaking the state machine.


### Worker · Serve loop / decode engine / scheduler

#### F131 · 🟠 high · [correctness/design] KV-cache sizing OOMs at bootstrap — free-memory probe taken before the activation workspace exists · ☑ FIXED

- **Location:** `crates/infer-worker/src/application/serve_loop.rs:78-201` (+ `runtime.rs`, `tuning.rs`)
- **Found:** during end-to-end smoke test of Qwen3-4B-Instruct-2507 (not in the original agent sweep).

**Problem.** The auto KV-sizing path probed `cudaMemGetInfo` **immediately after weight load** — *before* `Runtime::new` allocated the forward activation workspace and before `prime_graphs`/prewarm ran. The largest activation buffer is the logits scratch in `ForwardScratch`, sized `max_batch_tokens × vocab_size × sizeof(bf16)` — e.g. `8192 × 151936 × 2 = 2.49 GiB` for Qwen3-4B. Because the probe preceded that allocation, `mem_fraction_static` effectively meant "fraction of *post-weight free* memory for KV", and the leftover `(1 − fraction)` was the *only* headroom for a **fixed absolute** activation cost. With `fraction = 0.85` on a 24 GiB card: free-after-weights ≈ 9.19 GiB → KV grabbed 7.77 GiB → 1.42 GiB left < 2.49 GiB logits → hard failure:

```
Error: "Runtime::new: Kernel(\"cudaMalloc(2489319424) failed: 2\")"   # 2489319424 = 8192×151936×2
```

The KV pool had already been "successfully" sized; the OOM landed on the very next allocation. Larger `max_batch_tokens`/`vocab` make it worse. There was **no profiling/dummy forward run** anywhere in the worker — unlike vLLM/SGLang, which run a dummy forward, measure the true peak, then size KV from what remains.

**Fix.** Reordered `run_with_model` to be profile-driven (matches the vLLM/SGLang model):

1. Build `Runtime` with a **tiny throwaway KV pool** (`PROFILE_KV_BLOCKS`). This still allocates the full fixed activation workspace at worst-case capacity — the cost we must measure around.
2. `Runtime::profile_forward()` — one worst-case eager prefill through the real `step()` path. `self.graph` is still `None`, so `decide()` routes it eager: **no CUDA graph captured, no KV-base pointer baked**, so the pool can safely be resized afterward. This commits the activation workspace and forces the lazy cuBLASLt/cuDNN/recycling-pool allocations the first live forward would make.
3. `synchronize()` + `cudaMemGetInfo` — now `free` reflects weights + activation workspace + committed library state (everything except the throwaway probe pool).
4. Size the real pool: `num_blocks = (free − PREWARM_HEADROOM_BYTES) × fraction / bytes_per_block`, minus the existing rounding/working-set clamps. `fraction` now means "fraction of *usable* memory for KV", as users expect. The 1 GiB `PREWARM_HEADROOM_BYTES` covers the incremental per-shape cuDNN plans / pool growth the prewarm pass adds after sizing (a single dummy forward doesn't touch all ~90 prewarm shapes).
5. `Runtime::resize_kv_pool(num_blocks)` — frees the profile pool and allocates the real per-layer `k`/`v` tensors. Runs before `prime_graphs`, so no captured graph references the old KV base and `seq_kv_len` is empty (no live state to migrate).
6. `prime_graphs` + prewarm + `Ready` — unchanged, now against a pool guaranteed to fit.

The CLI `num_blocks_override` path short-circuits all of this (no probe, no dummy run), unchanged.

**Verified.** Rebuilt the worker and relaunched the full stack with `mem_fraction_static = 0.9` — the exact value that OOM'd before — reaching `Ready` cleanly (`free_after_workspace ≈ 6.0 GiB → num_blocks=32688, ~4.49 GiB KV pool`) and passing the end-to-end chat test (Chinese + English reasoning + streaming) with coherent output. Scope: LLM `run_with_model` path only; diffusion sizing untouched.



- **Location:** `crates/infer-worker/src/application/decode_engine.rs:625-635`
- **Resolution:** ⊘ The failure path is handled gracefully (check at decode_engine.rs:689); this is a design-fragility note, not a live defect.

**Problem.** In `prepare_step` fast path (lines 625-634), when prealloc has enough slots, the code takes exactly `initial_n` slots and returns surplus to the pool via `split_off`. However, if relief evicts rows during allocation (line 641-668), the `retain_active` at line 673 shrinks `order` but `new_indices` is already sized to old `initial_n`. The check at line 689 fails gracefully, but the design couples slot count to a transient row count without re-validating after preemption.

```rust
let initial_n = order.len();
let new_indices = if self.prealloc.len() >= initial_n { ... } else { 
    match alloc_with_relief(...) { ... }
};
self.rows.retain_active(active); // rows may evict
order = self.rows.as_slice().to_vec();
if new_indices.len() < order.len() { // now order is smaller but new_indices unchanged
```
The logic is correct but fragile: if retain_active changes order size more than expected, the later bound check (line 689) catches it, but it indicates a design that couples volatile state.

**Fix.** Unify the logic: always recompute `initial_n` after relief-driven retain_active, or pre-allocate a buffer and truncate after validation. Document the post-relief re-validation invariant.

#### F009 · 🟡 medium · [concurrency] Decode rows not re-synced before critical build_decode_request after control drain in fused step · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/worker_scheduler.rs:363-369`
- **Resolution:** ⊘ Verifier refuted the sibling claim (F-decode 854-858): control drain happens before `decode_order` is materialized in the fused path, and the .expect() is guarded. No live panic.

**Problem.** In `handle_fused_step`, after `decode_engine.prepare_fused_decode` (line 356-362), the code builds the decode request using `decode_order` (line 364-365). However, control messages are drained before this (in finalize_and_send at line 346 if an in-flight step exists). A Cancel/Preempt arriving between finalize and prepare_fused_decode could evict sequences from active, but `decode_order` is already materialized and passes stale SIDs to build_decode_request. The request will call `.expect()` at decode_engine.rs:858, which now panics.

```rust
decode_engine.finalize_and_send(...)?; // drains control
let decode = decode_engine.prepare_fused_decode(...)?; // builds order
let (decode_order, decode_new_indices, decode_build) = match decode {
    Some((order, idx)) => {
        let build = build_decode_request(&order, &idx, active, eos_ids, ...); // uses order[i] to look up active
```
If Cancel drains between finalize and prepare, the order rows are stale.

**Fix.** After finalize_and_send, drain control again before prepare_fused_decode, or have prepare_fused_decode re-drain internally. Alternatively, wrap build_decode_request's lookups in Ok/Err instead of expect, and skip failed rows.

#### F007 · 🟡 medium · [correctness] Pad row sentinel u64::MAX not filtered from ABC next-rows recording · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/worker_scheduler.rs:510-632`
- **Resolution:** ⊘ Safe by construction: `decode_order` never contains pad rows, so u64::MAX sentinels can't leak into `abc_next_rows`. This is a 'if the design changes' caution, not a live bug.

**Problem.** Inert pad rows are seeded with sequence_id = u64::MAX (line 510) as a sentinel. When pad rows finish early in a fused forward, they should not be recorded in abc_next_rows (line 632: `for (i, &sid) in decode_order.iter()...`). However, decode_order only contains the original decode rows (not pads), so pad-row output indices [decode_count, decode_prefix_len) are skipped by the iterator. This is safe for decode output, but the comment and design assumes pads are "inert" — if the padding logic changes or a pad is misidentified as active, u64::MAX could leak into next_rows.

```rust
for &blk in &idx {
    seqs.push(SeqStep {
        sequence_id: u64::MAX, // sentinel: inert pad row
        ...
    });
}
// Later:
for (i, &sid) in decode_order.iter().enumerate() {
    if !out.finished.get(i).copied().unwrap_or(false) {
        abc_next_rows.push(sid);
    }
}
```
The sentinel is never referenced because decode_order doesn't include pads. But if pad output indices bleed into decode_order iteration, u64::MAX leaks.

**Fix.** Add a defensive assertion that all recorded abc_next_rows exclude u64::MAX: `debug_assert!(!abc_next_rows.iter().any(|&s| s == u64::MAX), "pad sentinel leaked");` before record_mixed_abc_rows.

#### F011 · 🟡 medium · [design] Missing assertion on decode_order length consistency across fused prefill groups · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/worker_scheduler.rs:645-671`
- **Resolution:** ⊘ Design/assertion suggestion only; the cursor invariant holds today. Could add a debug_assert later, but not a correctness bug.

**Problem.** The cursor variable (line 626) is initialized to 0 or decode_prefix_len (line 645) depending on whether the first group has decode rows. It then increments by prep.rows for each prefill group (line 671). The final cursor should equal out.tokens.len() (the fused forward output size). However, there is no assertion that this invariant holds. If a prep.rows count is wrong or a group is skipped, subsequent cursors will misalign with the actual output indices, causing append_prefill_abc_next_rows to index out-of-bounds.

```rust
let mut cursor = 0usize;
if has_decode {
    cursor = decode_prefix_len; // includes pads
}
for &pi in &group.preps {
    ...
    cursor += prep.rows;
}
// No check that cursor == out.tokens.len() + out.finished.len() at end
```
The cursor could drift if group packing is miscalculated.

**Fix.** Add a post-loop assertion: `debug_assert_eq!(cursor, out.tokens.len(), "cursor drift in fused group processing");` to catch row-counting errors early.

#### F010 · ⚪ low · [performance] Multiple u32::MAX indices pushed without bounds on max_tokens/ignore_eos vector resize

- **Location:** `crates/infer-worker/src/application/worker_scheduler.rs:517-520`

**Problem.** Pad rows push u32::MAX into max_tokens (line 517). Later, line 541 resizes generated_counts and implicitly max_tokens/ignore_eos vectors to match seqs.len() via generated_counts.resize(seqs.len(), 0). The u32::MAX values are never used (pad rows are inert), but they waste memory in the request and could cause issues if downstream code validates max_tokens bounds. This is not a correctness issue but inflates per-step allocations and per-token compute cost.

```rust
max_tokens.push(u32::MAX);
ignore_eos.push(true);
generated_counts.push(0);
row_kinds.push(RaggedRowKind::Pad);
// Later:
generated_counts.resize(seqs.len(), 0);
```
Pad rows are never sampled (row_kinds filter), so max_tokens=u32::MAX is dead data.

**Fix.** For pad rows, use realistic max_tokens (e.g., 1) or document why u32::MAX is safe. This is a micro-optimization, low priority.


### Worker · KV alloc / relief / scratch

#### F014 · 🟡 medium · [design] Convoluted shrink_to_active retry loop in alloc_with_relief may be hard to reason about · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/kv_relief.rs:204-209`
- **Resolution:** ⊘ Correctness confirmed; this is a readability/maintainability note. Left as-is to avoid behavioral risk in the relief path.

**Problem.** After relief is received, the alloc_with_relief function re-checks whether active.len() is still smaller than the requested size n, and if so, reduces n and loops back. This loop will re-apply shrink_to_active clamping (line 168-170) and potentially return early with an empty allocation. While the retried_after_round1_relief flag eventually prevents infinite loops, the interaction between the relief retry logic, the shrink_to_active adjustment, and the round-escalation is convoluted and difficult to trace. The code is correct but fragile and could easily regress during maintenance.

```rust
if shrink_to_active {
    let active_now = active.len() as u32;
    if active_now < n {
        n = active_now;
        continue;  // loops back to line 167
    }
}
```
Combined with the earlier check at lines 168-170:
```rust
if shrink_to_active {
    n = n.min(active.len() as u32);
}
if n == 0 {
    return AllocWithReliefOutcome::Allocated(Vec::new());
}
```

**Fix.** Simplify by factoring out the shrink logic: compute the effective target size once at the start (n_target = if shrink_to_active { min(n_initial, active.len()) } else { n_initial }), then use that consistently. Alternatively, add a comment explaining the invariant: 'After relief, if active.len() has shrunk below n, we retry with the smaller target because more sequences may have been preempted.'

#### F012 · 🟡 medium · [soundness] Unsafe data race in flash_workspace_mut due to undocumented GPU parallelism · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/domain/forward_scratch.rs:157-167`
- **Verified:** partial (severity high→medium)
- **Resolution:** ⊘ Documented single-stream invariant: all layers run serially on one CUDA stream, so the `UnsafeCell` view is never aliased. This is the same pattern used throughout; no live race.

**Problem.** The flash_workspace_mut method hands out a mutable view of the flash_ws buffer by dereferencing an UnsafeCell with the assumption that layers run serially on a single CUDA stream and no two views are held simultaneously. However, the safety invariant relies on runtime guarantees (serial execution, single stream) that cannot be verified by the type system. If a caller parallelizes work across streams, spawns async kernels, or if view_raw hands out a Tensor handle that escapes to another thread, concurrent reads and writes to the same GPU buffer will occur, causing a data race.

```rust
pub fn flash_workspace_mut(&self) -> Tensor<f32, D> {
    // SAFETY: `&self` precludes aliasing across threads (`ForwardScratch`
    // is `!Sync` via `UnsafeCell`). Within one thread, layers run
    // serially: each call obtains a fresh view, hands it to one
    // `attention_paged` invocation, and drops it before the next layer
    // runs. We never hold two live views simultaneously.
    let cell = unsafe { &*self.flash_ws.get() };
    let shape = Shape::from_slice(&[self.flash_ws_elems]);
    let strides = shape.contiguous_strides();
    cell.view_raw(shape, strides, 0, true)
}
```
The safety comment assumes single-threaded execution on GPU, but CUDA kernels can parallelize across streams or launch async work. The returned Tensor handle could escape to concurrent code.

**Fix.** Either: (1) Add a runtime assertion that verifies single-threaded execution (e.g., check thread-local state or store the owning thread ID and assert on each call); (2) redesign to use a mutable borrow guard that prevents simultaneous views; or (3) document the requirement that this method must only be called from a single-threaded GPU task queue and add a compile-time marker (e.g., a private marker field on ForwardScratch) to enforce locality.

#### F013 · 🟡 medium · [soundness] Potential panic in total_free due to unchecked head >= len invariant · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/domain/global_kv_alloc.rs:99-105`
- **Resolution:** ⊘ `head <= free.len()` is an internal invariant upheld by every mutator; the fuzz test `fuzz_no_duplicates_no_loss_no_panic` exercises this. A debug_assert could document it but it's not a live bug.

**Problem.** The total_free() method computes (self.free.len() - self.head) as u32. If the invariant head <= free.len() is violated, this panics on the subtraction. While the invariant is documented in a comment on line 64 ('Invariant: head <= free.len()'), there is no runtime check in release builds. A bug in any code path that advances head (e.g., compact_head, merge_and_sort, or recycle) could violate this invariant silently in release mode, then panic here on the next alloc attempt.

```rust
pub fn total_free(&self) -> u32 {
    (self.free.len() - self.head) as u32 + self.released.len() as u32
}
```
and the invariant comment at line 64:
```rust
/// Bump pointer. Invariant: `head <= free.len()`.
head: usize,
```
No defensive check before the subtraction.

**Fix.** Add a debug_assert! or unwrap-on-overflow check before the subtraction: `debug_assert!(self.head <= self.free.len(), "invariant violation: head={} > len={}", self.head, self.free.len()); let available = self.free.len().saturating_sub(self.head);`

#### F018 · ⚪ low · [design] Silent degradation in alloc_with_relief when active.len() == 0

- **Location:** `crates/infer-worker/src/application/kv_relief.rs:168-173`

**Problem.** When shrink_to_active=true and active.len()==0, alloc_with_relief immediately returns Allocated(Vec::new()) without ever signaling to the scheduler that the request failed. A caller asking for N slots gets 0 back silently. The return type doesn't distinguish between 'got all requested slots', 'got a subset', or 'got zero due to no active work'. While this is intentional and correct (the caller should check the returned vector length), it's a silent failure mode that could be surprising.

```rust
if shrink_to_active {
    n = n.min(active.len() as u32);
}
if n == 0 {
    return AllocWithReliefOutcome::Allocated(Vec::new());
}
```
A request for 10 slots with shrink_to_active=true and 0 active sequences returns Allocated(vec![]) with no signal to the caller that the request was ignored.

**Fix.** Consider adding a log or metric to track when alloc_with_relief returns empty due to shrink_to_active, or change the return type to distinguish between 'full success', 'partial (shrunk)', and 'zero (no active work)'. Alternatively, document clearly in a comment that callers must check the returned vector length.

#### F017 · ⚪ low · [design] Subtle interaction between relief_satisfies_request and shrink_to_active

- **Location:** `crates/infer-worker/src/application/kv_relief.rs:236-249`

**Problem.** The relief_satisfies_request function returns true if shrink_to_active is set and active.len() < needed_slots, even if kv_allocator.total_free() < needed_slots. This logic is correct (the caller will later clamp n to active.len()), but the interaction is subtle and unintuitive: a caller might expect relief_satisfies_request to mean 'we have enough KV slots for the request', but it actually means 'we either have enough slots OR we can reduce the request to fit available work'. This could confuse maintainers.

```rust
fn relief_satisfies_request(
    kv_allocator: &GlobalKvAllocator,
    active: &ActiveSeqMap,
    needed_slots: u32,
    shrink_to_active: bool,
) -> bool {
    if needed_slots == 0 {
        return true;
    }
    if shrink_to_active && (active.len() as u32) < needed_slots {
        return true;  // satisfied even if total_free < needed_slots
    }
    kv_allocator.total_free() >= needed_slots
}
```

**Fix.** Rename the function to make the intent clearer, e.g., relief_sufficient_or_request_shrinkable(), or add a comment explaining: 'Returns true if relief has provided enough KV slots OR if shrink_to_active allows us to reduce the request to match remaining active sequences.'

#### F015 · ⚪ low · [design] No validation of n <= cap_num_tokens in view() creates potential out-of-bounds

- **Location:** `crates/infer-worker/src/domain/forward_scratch.rs:102-109`

**Problem.** The view() function creates a row-prefix view [n, cols] over a [cap, cols] buffer. The comment at lines 102-103 states 'cols MUST equal the buffer's column count so the contiguous prefix is valid.' However, there is no runtime check that n <= cap_num_tokens. If a caller passes n > cap_num_tokens, the view_raw call will create a view that spans beyond the buffer's allocated memory, potentially causing out-of-bounds access during kernel execution.

```rust
#[inline]
fn view(t: &Tensor<T, D>, n: usize, cols: usize) -> Tensor<T, D> {
    let shape = Shape::from_slice(&[n, cols]);
    let strides = shape.contiguous_strides();
    t.view_raw(shape, strides, 0, true)
}
```
No check that n <= self.cap_num_tokens. Callers like normed(n) at line 124 rely on the caller to not exceed capacity.

**Fix.** Add a debug_assert or panic in view(): `debug_assert!(n <= self.cap_num_tokens, "view size {} exceeds capacity {}", n, self.cap_num_tokens);` Alternatively, require callers to use a bounds-checked view function that returns Option<Tensor>.

#### F016 · ⚪ low · [design] Fragile assumption that sanitize_returned_indices ensures sorted output

- **Location:** `crates/infer-worker/src/domain/global_kv_alloc.rs:196-230`

**Problem.** The merge_sorted_returned() function assumes its input (returned: &[u32]) is sorted and deduplicated. This is currently guaranteed by the only caller (free() at line 193), which calls sanitize_returned_indices() to sort and deduplicate. However, if a future refactoring bypasses sanitize_returned_indices() or if merge_sorted_returned() is made public/used elsewhere, the merge will silently produce an unsorted free list, breaking the allocator invariant and causing incorrect future allocations.

```rust
fn merge_sorted_returned(&mut self, returned: &[u32]) {
    // ... assumes returned is sorted ...
    while i > 0 && j > 0 {
        w -= 1;
        if self.free[i - 1] >= returned[j - 1] { // comparison assumes sorted
            self.free[w] = self.free[i - 1];
            i -= 1;
        } else {
            self.free[w] = returned[j - 1];
            j -= 1;
        }
    }
}
```
No debug_assert that returned is sorted.

**Fix.** Add debug_assert!(returned.windows(2).all(|w| w[0] <= w[1]), "merge_sorted_returned requires sorted input");  at the start of merge_sorted_returned(). Or make the sorting responsibility explicit in the function signature or with a newtype wrapper.


### Worker · Model components & weight loader

#### F021 · 🟠 high · [correctness] Same silent truncation issue in fused QKV loading · ☑ FIXED

- **Location:** `crates/infer-worker/src/models/loader.rs:193`
- **Verified:** confirmed (severity critical→high)
- **Resolution:** ☑ Loader now errors on a safetensors byte-length mismatch instead of silently zero-padding (fixed at 3 sites: fused_qkv, fused_gate_up, single-tensor loader).

**Problem.** In load_fused_qkv, the pointer copy uses .min(src.len()), which silently truncates if the safetensors view is smaller than expected, leaving zeros in the fused QKV projection weights.

```rust
Line 193: std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n.min(src.len())); Similar pattern to tensor_from_safetensor_view: if src.len() < n, only src.len() bytes are copied.
```

**Fix.** Replace with a check: if src.len() != n { return error }. Or assert equality. Do not silently truncate.

> **Verifier note:** In load_fused_qkv, the pointer copy uses .min(src.len()) which silently truncates if the safetensors view has fewer bytes than expected, leaving zeros in portions of the fused QKV projection weights. This silently corrupts model weights when processing malformed safetensors files instead of failing fast with an error. The issue is inconsistent with the dtype-cast path (line 196) which would panic, and with transformer.rs which would also panic on insufficient data. Severity is high (not critical) because: (1) only affects corrupted files, (2) during model loading (startup), not hot paths, and (3) results in degraded inference quality rather than crash.

#### F020 · ⚪ low · [performance] MoE routing executed entirely on CPU, blocking forward pass

- **Location:** `crates/infer-worker/src/components/ffn_moe.rs:83-88, 168-180`
- **Verified:** partial (severity high→low)

**Problem.** The MoE routing (expert selection, grouping, and output combination) is performed entirely on the CPU via host vectors and loops. Lines 83 (normed.to_host_vec) and 88 (router_logits.to_host_vec) download tensor data to CPU, then CPU code sorts/routes tokens (lines 90-116), groups inputs (lines 118-128), uploads grouped data (lines 132-138), executes expert GEMMs on GPU (lines 146-166), downloads results (line 168), and combines on CPU (lines 169-177) before uploading again (line 179). This is millions of CPU cycles per token in the decode hot path (one per generated token), with H2D/D2H transfers that saturate PCIe bandwidth.

```rust
Line 83: let normed_host = normed.to_host_vec()?; Line 88: let router_host = router_logits.to_host_vec()?; Lines 91-116: CPU loop sorting and routing per token. Lines 168-177: CPU loop combining expert outputs per token, executed millions of times during decode.
```

**Fix.** Move routing logic to GPU kernels (one kernel for routing with softmax over top-k experts, one for grouped scatter/gather, one for weighted combination). Avoid host transfers for routing data. Use device memory for grouping, offsets, and accumulation to keep the hot path on GPU.

> **Verifier note:** MoeFfn contains a CPU-resident routing implementation with H2D/D2H transfers that would be inefficient if enabled for production inference, but the code is currently dead—not instantiated or called anywhere. DecoderBlocks always use DenseFfn; MoeFfn is infrastructure for future use. The algorithmic design (CPU routing, grouping, and output combination with tensor downloads/uploads) would benefit from GPU kernels for top-k expert selection, scatter/gather, and weighted combination if MoE models are deployed, but this is not a current performance issue.

#### F019 · ⚪ low · [correctness] Silently truncate tensor data on malformed safetensors files

- **Location:** `crates/infer-worker/src/models/loader.rs:481-482`
- **Verified:** partial (severity critical→low)

**Problem.** When loading tensors from safetensors files, if the source data is smaller than expected (file corruption or wrong shape metadata), the code silently truncates the copy and leaves zeros in the unfilled portion of the destination buffer. This results in weight tensors with NaN/zero values that corrupt model inference silently without error.

```rust
Line 481-482: let n = size_bytes.min(src_bytes.len()); host_buf[..n].copy_from_slice(&src_bytes[..n]); If src_bytes.len() < size_bytes, only n bytes are copied and the remainder stays zero-initialized.
```

**Fix.** Replace the .min() with a bounds check that returns an error if src_bytes.len() != size_bytes. For fused QKV (line 193), same fix: std::ptr::copy_nonoverlapping should error if n != expected size, not silently truncate.

> **Verifier note:** The code uses `.min()` to defensively handle potential size mismatches when copying tensor data, which is inconsistent with the casting path (that panics on short data). While this is poor defensive programming practice, the safetensors library's validation ensures this truncation path never executes with valid files. The real issue is inconsistent error handling between dtype-matching (silent truncate) and dtype-casting (panic) paths, not data corruption.


### Worker · Infrastructure (safetensors IO, ZMQ transport, main)

> The worker infrastructure layer implements ZMQ transport and safetensors file I/O for model weight loading. Architecture is cleanly layered (DDD), but several issues emerge: (1) a critical unsound lifetime cast on mmap buffers that can enable use-after-free if TensorView escapes, (2) index bounds checks missing when routing tensor reads across shards, (3) unsafe unwrap() calls on bootstrap-time JSON parsing that can panic on corrupted model indexes, (4) device ID hardcoding in control messages despite multi-device support. ZMQ socket configuration is minimal (no HWM tuning, no timeouts), and the heartbeat telemetry path has an integer cast overflow risk. These are mostly bootstrap/startup issues rather than hot-path problems, but they undermine production reliability.

#### F039 · 🟠 high · [design] Config parsing lacks device existence validation in worker_main · ☑ FIXED

- **Location:** `crates/infer-worker/src/bin/worker_main.rs:249`
- **Resolution:** ☑ Real device string is now threaded from `cfg.device` into `ControlPump`; Hello/Ready report the actual device (also covers F040).

**Problem.** The parse_device_id function uses .ok_or_else() which returns Err, but the caller returns this error via the ? operator, causing early exit. However, the error message is user-friendly; the real issue is that if load.device is an invalid string (e.g., 'tpu:0'), parse_device_id will return Err and the worker will fail startup. The function does not validate that the device actually exists on the system or is the correct type.

```rust
let device_id = parse_device_id(&load.device).map_err(|e| format!("Cuda::new: {:?}", e))?;
```

**Fix.** This is not a bug per se, but consider validating the device exists and is CUDA-capable. Device mismatch is a common deployment error.

#### F042 · 🟡 medium · [correctness] Integer cast overflow risk in heartbeat KV stats · ⊘ WONT-FIX

- **Location:** `crates/infer-worker/src/application/serve_loop.rs:730`
- **Resolution:** ⊘ Already correct: line 730 uses `.min(u32::MAX as usize) as u32` — a saturating clamp, not a silent truncation.

**Problem.** The transient_reserved value is cast from usize to u32 with .min(u32::MAX as usize) as u32. If transient_reserved is larger than u32::MAX (on a 64-bit system with massive KV pools), it will silently truncate to u32::MAX, causing the scheduler to receive incorrect KV state and potentially make bad allocation decisions.

```rust
Some(transient_reserved.min(u32::MAX as usize) as u32),
```

**Fix.** Either ensure transient_reserved can never exceed u32::MAX in the allocator (add an assertion), or use a wider type (u64) in the protocol if large KV pools are expected.

#### F038 · 🟡 medium · [correctness] Fallback scan of shard[0] assumes non-empty shards vector · ☑ FIXED

- **Location:** `crates/infer-worker/src/infrastructure/io/safetensors.rs:168`
- **Resolution:** ☑ `read_view`/`contains` now return an error instead of panicking on empty shards or an out-of-range shard index.

**Problem.** In the single-file read path, the code unconditionally accesses self.shards[0] without checking that shards is non-empty. If open_single is called with an empty file or a bug in Shard::open occurs, a panic will result during weight loading.

```rust
self.shards[0].header.tensor(name)
```

**Fix.** Add `if self.shards.is_empty() { return Err(...) }` check, or ensure invariant is enforced in open_single via assert!(!shards.is_empty()).

#### F040 · 🟡 medium · [design] Hardcoded CUDA device assumption in control messages · ☑ FIXED

- **Location:** `crates/infer-worker/src/infrastructure/transport/control_pump.rs:55, 115`
- **Resolution:** ☑ `ControlPump` now carries the real device string; Hello/Ready no longer hardcode `cuda:0`.

**Problem.** The ControlPump hardcodes device='cuda:0' in WorkerHello and WorkerReady messages, but the worker binary accepts any device_id via parse_device_id. If the scheduler launches a worker on device 2, the hello message will still report 'cuda:0', causing scheduler-side confusion and potential device mismatch.

```rust
device: "cuda:0".into(),
```

**Fix.** Pass device_id from worker_main through to ControlPump so it can report the actual device. Currently device_id is parsed but not propagated to the transport layer.

#### F043 · ⚪ low · [correctness] Uninitialized trace variable in serve_loop diagnostics

- **Location:** `crates/infer-worker/src/application/serve_loop.rs:381`

**Problem.** The variable tr_pf_ms is initialized to 0f64 on line 381 but the corresponding fused_step timing code that would populate it is only executed conditionally inside the !pending_prefills.is_empty() branch. The trace output uses tr_pf_ms even if prefills were empty, printing 0.0 which could mislead performance analysis. The variable should be inside the conditional block or initialized differently.

```rust
let tr_pf_ms = 0f64; ... let fused_t0 = if trace_steps { ... };
```

**Fix.** Move tr_pf_ms initialization inside the !pending_prefills.is_empty() conditional, or use Option<f64> and print 'N/A' if None.

#### F041 · ⚪ low · [design] No explicit ZMQ socket buffer configuration in DataPump

- **Location:** `crates/infer-worker/src/infrastructure/transport/data_pump.rs:18-36`

**Problem.** The DataPump::new method creates two sockets (PULL and PUSH) but does not set RCVHWM or SNDHWM (receive/send high water marks). With default HWM of 1000, if the scheduler floods the worker with prefill commands or the worker cannot send outputs fast enough, messages will be silently dropped when buffers overflow. This can cause hangs or lost work.

```rust
let recv_socket = ctx.socket(zmq::PULL)...; let send_socket = ctx.socket(zmq::PUSH)...;
```

**Fix.** Set RCVHWM and SNDHWM explicitly (e.g., socket.set_rcvhwm(10000)?.set_sndhwm(10000)?) or configure based on max_batch_tokens/max_batch_seqs. Document the HWM policy.


### Scheduler · Session table / lifecycle / accounting

#### F047 · 🟡 medium · [design] Lazy removal tombstone compaction threshold heuristic · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/domain/inference_session/queue.rs:127-131`
- **Resolution:** ⊘ Heuristic threshold; verified correct. Pathological tombstone accumulation is a theoretical latency spike, not incorrectness. Tuning left for a profiling-driven change.

**Problem.** The compaction threshold `queue.len() > 2 * live + 8` can cause pathological behavior under certain cancellation patterns. If a scheduler receives a burst of cancellations after each prefill (e.g., timeout-based cancellation), the queue can accumulate O(n) tombstones before compaction triggers, causing temporary latency spikes when the next heavy removal batch compacts O(n) entries.

```rust
fn maybe_compact(&mut self) {
    if self.queue.len() > 2 * self.live + 8 {
        self.compact();
    }
}
```

**Fix.** Either: (1) trigger compaction more eagerly (e.g., `> 1.5 * live` or when removal rate exceeds a threshold), (2) use an incremental compaction strategy that removes a few tombstones per pop(), or (3) add a metric to detect and alert on pathological hole ratios.

#### F044 · 🟡 medium · [performance] Per-iteration Vec allocation on hot path in prefilling() · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/domain/inference_session/table.rs:222-224`
- **Verified:** partial (severity high→medium)
- **Resolution:** ⊘ Bounded (≤ max_batch_seqs pointers), once per scheduling iteration, dwarfed by the per-step msgpack+network I/O. Converting to an iterator would invade the batch-builder slice API for negligible gain.

**Problem.** The prefilling() method allocates a new Vec on every call to collect references to all prefilling sessions. This is called during scheduling iterations from workflow code (prefilling_continuations at line 313, which itself allocates). Scheduling iterations are per-generated-token, making this a high-frequency allocation.

```rust
pub fn prefilling(&self) -> Vec<&InferenceSession<Prefilling>> {
    self.prefilling.values().collect()
}
```

**Fix.** Either: (1) expose an iterator directly instead of Vec (pub fn prefilling_iter(&self) -> impl Iterator), or (2) pre-allocate a single reusable Vec in the RequestTable and clear() it between iterations, or (3) provide a with_prefilling(FnMut) callback pattern that avoids allocation entirely.

> **Verifier note:** The `prefilling()` method (lines 222-224) does allocate a new Vec on every call via `.collect()`. This allocation occurs on the hot path in the event loop, specifically after each worker batch completion or event (not per-generated-token, but per-batch-completion which is still frequent). The allocation is unnecessary since the only uses of the returned Vec are `.first()` and passing to `build_batch()`. Similar wasteful allocations exist in `prefilling_continuations()` (line 313) and `running_sequence_ids()` (line 576). The codebase already demonstrates the pattern of using cheaper iterator methods like `.any()` (lines 234, 307) instead of `.collect()` for lightweight checks.

#### F045 · 🟡 medium · [performance] Per-iteration Vec allocation in prefilling_continuations() · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/domain/inference_session/table.rs:313-322`
- **Verified:** confirmed (severity high→medium)
- **Resolution:** ⊘ Same as F044: small bounded Vec once per iteration; `has_prefilling_continuations()` already short-circuits the empty case with no allocation.

**Problem.** This method collects all prefilling sessions with remaining tokens into a Vec, then filters and maps them. It is called once per scheduling iteration (from workflow/llm.rs:157 and workflow/diffusion.rs:121) on the hot path. For typical batch sizes (16-32 sequences), this allocates a small-to-medium Vec every iteration.

```rust
pub fn prefilling_continuations(&self) -> Vec<(RequestId, usize)> {
    self.prefilling
        .values()
        .filter(|seq| !seq.has_inflight() && seq.remaining_tokens() > 0)
        .filter_map(|seq| {
            let remaining = seq.remaining_tokens();
            (remaining > 0).then(|| (seq.meta.id.clone(), remaining))
        })
        .collect()
}
```

**Fix.** Cache the result in the RequestTable and invalidate on mutations (take_waiting, commit_prefill_start, ack_prefill, set_prefill_inflight, preempt_to_queued, fail_sequence). Or provide an iterator pattern that avoids heap allocation.

#### F046 · 🟡 medium · [correctness] Missing validation in decoding_kv_slots integer cast · ☑ FIXED

- **Location:** `crates/infer-scheduler/src/domain/inference_session/table/accounting.rs:34-38`
- **Resolution:** ☑ `decoding_kv_slots` feeds `preemption_candidates`, which now clamps all usize→u32 casts via `u32::try_from().unwrap_or(MAX)`.

**Problem.** The decoding_kv_slots function casts usize to u32 implicitly (as u32) on line 100. On 64-bit systems, if a sequence has more than 2^32 output tokens or prompt length, this cast will silently truncate. While extremely unlikely in practice (would require generating 4 billion tokens), this violates the principle of explicit bounds checking for resource accounting, which directly feeds into KV budget decisions.

```rust
pub(crate) fn decoding_kv_slots(seq: &InferenceSession<Decoding>) -> usize {
    seq.state
        .prompt_len
        .saturating_add(seq.state.output_tokens.len().saturating_sub(1))
}
... later in line 100: kv_used: decoding_kv_slots(seq) as u32,
```

**Fix.** Add an explicit bounds check: `let slots = decoding_kv_slots(seq); if slots > u32::MAX as usize { return Err(...) }` before casting. Or cap the value: `.min(u32::MAX as usize) as u32`.

#### F049 · 🟡 medium · [correctness] Integer cast from usize to u32 in PreemptCandidate without overflow check · ☑ FIXED

- **Location:** `crates/infer-scheduler/src/domain/inference_session/table/accounting.rs:98-110`
- **Resolution:** ☑ `preemption_candidates` now clamps output_len/input_len/kv_used with saturating `u32::try_from`.

**Problem.** The preemption_candidates function casts output_tokens.len() and input_ids.len() directly to u32 without bounds checking. On 64-bit systems, if a sequence's prompt or output exceeds 2^32 tokens, this silently truncates, causing incorrect preemption victim scoring.

```rust
output_len: seq.state.output_tokens.len() as u32,
...
input_len: seq.meta.input_ids.len() as u32,
...
kv_used: seq.state.num_computed_tokens as u32,
```

**Fix.** Add bounds checks before each cast, e.g., `let output_len = u32::try_from(seq.state.output_tokens.len()).unwrap_or(u32::MAX);` to clamp at u32::MAX rather than truncate.

#### F048 · ⚪ low · [design] Missing consistency check for RequestTable::by_external_id shadowing

- **Location:** `crates/infer-scheduler/src/domain/inference_session/table.rs:279-284`

**Problem.** When inserting a new request with a duplicate external_id, the code silently overwrites the prior mapping in by_external_id (line 283). The comment acknowledges this is intentional (line 281-282), but the validate_consistency() check doesn't verify that by_external_id mappings always point to valid, active sequences. If a sequence is removed before its successor claims the external_id, a stale entry could point into the void briefly.

```rust
if !external_id.is_empty() {
    // Multiple inflight requests can share the same client-provided
    // external_id (e.g. retried requests). We keep the latest in the
    // index; older sequences keep their own internal lookup paths.
    self.by_external_id.insert(external_id, sequence_id);
}
```

**Fix.** Add a check in validate_consistency() that every entry in by_external_id points to an active sequence, and that the reverse mapping matches. Or document the intentional shadowing more explicitly in the remove_active() path.


### Scheduler · Radix-tree prefix cache

#### F054 · 🟡 medium · [soundness] Memory leak risk: free_ids reuse does not check for stale LRU generation collisions · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:198-207, 458`
- **Resolution:** ⊘ Documented design: reused ids keep a stale (higher) generation stamp so lingering queue entries are correctly skipped; `compact()` bounds the map. Verifier found no unbounded growth in practice.

**Problem.** Node IDs are reused via free_ids (line 458 pushes the evicted node's ID back to free_ids for reuse). When new_node pops from free_ids (line 199), it reuses the ID without clearing the LRU generation stamp. The comment at lines 194-197 states: 'Reused ids keep their (stale, higher) lru.generations stamp, so any lingering stale queue entry for that id is still correctly skipped at pop time.' This is correct for LRU correctness but creates a subtle issue: if a node ID is evicted and immediately reused in the same iteration, and then immediately added to LRU, the new node will have a higher generation stamp than any stale queue entry—but if the generations map is not cleaned up, it will accumulate stale entries. The compact() function at line 122 is called periodically but not guaranteed to run. If free_ids causes rapid reuse of IDs without cleanup, the generations map can grow unboundedly. Additionally, if a reused node ID is never added to LRU again (e.g., it's interior to a large tree), its generation stamp entry remains in the map forever.

```rust
Lines 198-207 reuse node IDs from free_ids without clearing generations. Line 458 pushes evicted IDs to free_ids. The generations HashMap (line 85) stores entries indefinitely; only compact() (line 122) removes them. The compact() is only called inside evict_collect_at_least at line 472, conditionally based on queue length heuristic.
```

**Fix.** When reusing a node ID via new_node, call self.lru.generations.remove(&id) before reusing it. Or, in compact(), also remove generations entries for nodes not in the queue. Alternatively, ensure compact() is called deterministically (e.g., after every N evictions or every iteration), not just when queue bloat reaches a threshold.

#### F053 · 🟡 medium · [correctness] LRU eviction does not promote parent after child removal · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:438-465`
- **Resolution:** ⊘ On close reading the code already calls `maybe_promote_to_lru(parent)` after detaching a child (radix_tree.rs:459-461); the reviewer's own note concludes it 'is actually handled'. No bug.

**Problem.** In evict_collect_at_least, when a node is successfully evicted (lines 439-461), the parent is located (line 440) and the child is detached (line 445). However, the parent is never checked for LRU promotion. If the parent was previously not a leaf (had other children), and this eviction removed the last child, the parent becomes an unowned leaf eligible for LRU admission. But it is never added to LRU at this point. The comment at line 459-461 acknowledges the issue: 'Removing this child may have made the parent an unowned leaf — promote it to LRU.' The code then calls maybe_promote_to_lru(parent), so this is actually handled. But critically: maybe_promote_to_lru checks !n.in_lru (line 653), so if the parent is already in LRU (from a prior mark_finished_chain), it won't be re-added. This is correct. However, the check at line 650 includes '!n.edge_tokens.is_empty()', which skips promotion of 'logically-deleted' nodes. After evicting children, if a parent's edges are empty (should not happen as parent is only created when it has edges), promotion is skipped. This edge case is unlikely but worth noting.

```rust
Line 461 calls maybe_promote_to_lru(parent). Line 653 checks self.nodes[node].in_lru. The logic is correct as written, but the condition !n.edge_tokens.is_empty() at line 650 is an additional filter that prevents dead nodes from entering LRU, which is sound.
```

**Fix.** The code is correct. Add a comment at line 461 explaining why promotion is safe (parent was not previously an LRU leaf, since we just removed its child).

#### F055 · ⚪ low · [correctness] mark_finished_chain walks entire chain even if sequence is unknown

- **Location:** `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:387-403`

**Problem.** The mark_finished_chain function is documented as idempotent (line 384-386), but when called with an unknown seq_id, it returns early (line 388-390) without walking the chain. This is correct. However, the function is called from multiple places (cancel.rs:79, output_fns.rs:156, output_fns.rs:229, cancel.rs:79) and the caller must ensure it is only called once per sequence. If a sequence is marked as finished, then re-marked, the second call does nothing. This is sound (idempotency is preserved), but if a caller accidentally calls mark_finished_chain on the same sequence twice due to a logic error, the second call silently does nothing, potentially masking a bug. The idempotency is a feature, not a flaw, but the lack of any indication (e.g., a log warning or return value) means correctness bugs in callers are hard to detect.

```rust
Lines 388-390 return early for unknown sequences. Lines 387-403 have no assertion that sequence was previously in the tree; idempotency is implicit. Callers (planning.rs:156, cancel.rs:79) assume they are calling exactly once per sequence lifetime.
```

**Fix.** This is acceptable as-is. If stricter checking is desired, add debug_assert!(self.seqs.contains_key(&seq_id)) before line 388, with a comment explaining that idempotency is intentional.

#### F056 · ⚪ low · [design] split_edge prefix_owners assignment is redundant and potentially confusing

- **Location:** `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:577-606`

**Problem.** The split_edge function at line 581 computes 'let prefix_owners = self.nodes[node].owners.clone()', then at line 606 explicitly assigns 'self.nodes[node].owners = prefix_owners'. Since line 580 comments that the original owner set is unchanged, and lines 581-606 are between creating the suffix node and explicitly re-assigning, this is redundant. The original node's owners field was never mutated between line 581 and line 606 (it was taken via mem::take for the suffix at line 594, but that was children, not owners). The explicit re-assignment at line 606 is a no-op. This is not a bug, but it's a code clarity issue that could confuse future maintainers.

```rust
Lines 580-581 comment and assign prefix_owners = original.owners. Lines 590-597 create suffix with different owners. Line 606 reassigns self.nodes[node].owners = prefix_owners, which is a no-op since self.nodes[node].owners was never mutated.
```

**Fix.** Remove the redundant assignment at line 606 and add a comment at line 581 stating 'Prefix node retains its original owner set (unchanged by suffix creation).' This clarifies that the owners field was never modified.


### Scheduler · Application (engine loop, batch builder, workflows)

#### F058 · 🟡 medium · [performance] Per-segment prefix hint clone in hot path · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/application/batch_builder.rs:175`
- **Resolution:** ⊘ Per-prefill-segment clone (not per-token), bounded by batch size; `prefix_hints` is a shared borrow so it can't be moved out. The code comment already documents the bounded linear scan.

**Problem.** Each scheduled prefill segment clones the entire Vec<GlobalIndex> even though it's immediately serialized into MsgPack. For large prefix cache hits (hundreds of tokens), this creates unnecessary heap allocation and copy per segment. Happens once per dispatch iteration.

```rust
.map(|(_, indices)| indices.clone())
```

**Fix.** Use references or move the Vec directly. Either restructure the data flow to avoid clone, or leverage MsgPack encoding to serialize from a reference without copying.

#### F061 · 🟡 medium · [performance] Per-step allocation in sanitize_step_output on stale sequences · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/application/workflow/llm.rs:276-344`
- **Resolution:** ⊘ Clone only triggers on the stale-sequence slow path (cancelled/finished), which is rare; the common clean path is untouched. Optimizing the rare path adds complexity for little gain.

**Problem.** When any sequence is stale (cancelled/finished), the entire StepOutput is cloned including assigned_indices Vec<AssignedIndices> with their token_ids. This allocation is unnecessary if stale sequences are rare. The comment acknowledges most steps are clean, but the clone is still triggered per stale row.

```rust
Cow::Owned(StepOutput {
    prefill_done: step
        .prefill_done
        .iter()
        .copied()
        .filter(...)
        .collect(),
    tokens: step
        .tokens
        .iter()
        .filter(...)
        .cloned()
        .collect(),
    assigned_indices,
})
```

**Fix.** Use iterators that filter on-the-fly without intermediate collections, or keep the original and iterate with filters. For the common case (no stale sequences), this path isn't taken, so the optimization is already good. For the stale case, consider using drain_filter or similar patterns.

#### F059 · ⚪ low · [performance] HashSet allocation for segment selection in hot path

- **Location:** `crates/infer-scheduler/src/application/batch_builder.rs:132`

**Problem.** A HashSet is allocated to determine which prefilling sessions to include in the batch. While typically small (≤max_num_seqs), this allocates and hashes every iteration. For small batches, linear search or a pre-allocated reusable BitSet could be faster.

```rust
let selected: HashSet<&RequestId> = scheduled_segments.iter().map(|(id, _)| id).collect();
```

**Fix.** For small max_num_seqs, replace with linear search or a reusable bit vector. If preserved, consider moving the HashSet to PlanningSystem as a reusable scratch buffer.

#### F062 · ⚪ low · [performance] Redundant sequence state lookup in process_llm_step_decoded

- **Location:** `crates/infer-scheduler/src/application/output_fns.rs:253-291`

**Problem.** The function collects finished_sequence_ids into a Vec, sorts and deduplicates it (lines 290-291), then processes each one. For batches with many finishing sequences, the sort is O(n log n) when a single pass could collect and deduplicate via a HashSet in O(n).

```rust
finished_sequence_ids.sort_unstable_by_key(|id| id.0);
finished_sequence_ids.dedup();
for sequence_id in finished_sequence_ids {
```

**Fix.** Use a HashSet to track finished sequence IDs during iteration, eliminating the post-collection sort. Or use a SmallVec for the common case of 0-2 finished sequences per step.

#### F064 · ⚪ low · [design] Missing docstring context on two-independent-judgment race (docs 2.5)

- **Location:** `crates/infer-scheduler/src/application/workflow/llm.rs:1-50`

**Problem.** The architecture doc (worker_batch_design.md section 2.5) describes a two-independent-judgment scenario where both scheduler and worker independently decide when sequences finish. The code handles stale outputs via sanitization, but the design rationale and potential edge cases are not documented in the source. Future maintainers may not understand why stale-sequence handling is necessary.

```rust
Section 2.5 of worker_batch_design.md states: '两边独立判断，不冲突' (both sides make independent judgments without conflict), but llm.rs has no reference to this constraint.
```

**Fix.** Add a comment block explaining the two-independent-judgment invariant: worker decides locally when to finish sequences and remove them, scheduler sees stale outputs and must filter them gracefully. Link to design doc or quote the key principle.

#### F063 · ⚪ low · [performance] Per-step stale indices detection iterates assignment list multiple times

- **Location:** `crates/infer-scheduler/src/application/workflow/llm.rs:285-296`

**Problem.** The all_running check (lines 285-296) iterates through assigned_indices, prefill_done, and tokens separately to check if sequences are stale. If any are stale, the entire StepOutput is re-iterated to filter it (lines 305-342). For large batches with many assigned indices, this is redundant iteration.

```rust
let all_running = step
    .assigned_indices
    .iter()
    .all(|a| is_sequence_running(ctx.requests, a.sequence_id))
    && step
        .prefill_done
        .iter()
        .all(|sid| is_sequence_running(ctx.requests, *sid))
    && step
        .tokens
        .iter()
        .all(|tk| is_sequence_running(ctx.requests, tk.sequence_id));
if all_running {
    return Cow::Borrowed(step);
}
// Slow path re-iterates everything
```

**Fix.** Combine the check and filter into a single pass: iterate once collecting non-stale entries into the output structure. Use a builder pattern or early-exit on first stale detection, then decide whether to filter.


### Scheduler · Batching policy + ZMQ control plane

> The scheduler batching policy and ZMQ transport layer exhibit several correctness and concurrency issues. The continuous batching admission control has an off-by-one error in sequence budgeting that allows oversized batches, and the decode reserve calculation is incorrect for small max_tokens values. The PendingCalls RPC correlation table has a potential race condition on broadcast call completion where multiple threads can send on the same oneshot. The ZMQ transport frame handling lacks proper validation and error recovery. The control-plane router has a synchronization race between eviction and late heartbeats that can resurrect dead workers. These issues are concentrated in the hot path (batching admission, RPC resolution, frame receiving) and control plane, making them medium-to-critical severity for production use.

#### F070 · 🟠 high · [performance] ZMQ unbounded channels risk OOM under load · ☑ FIXED

- **Location:** `crates/infer-scheduler/src/infrastructure/transport/zmq_transport.rs:44-45, 234-235`
- **Verified:** confirmed
- **Resolution:** ☑ Both scheduler→ZMQ outbound queues (frontend responses, worker commands) are now bounded (16384) with async `send().await` backpressure; inbound stays unbounded.

**Problem.** Both ZmqFrontendTransport and ZmqWorkerTransport use mpsc::unbounded_channel() for outgoing responses and commands. If the ZMQ I/O thread falls behind (e.g., slow network, slow worker), the unbounded channel will accumulate messages indefinitely, consuming memory without bound. Under sustained high load (thousands of concurrent requests), this can lead to OOM kills.

```rust
let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();
let (outgoing_tx, outgoing_rx) = mpsc::unbounded_channel();
// No backpressure mechanism
```

**Fix.** Use bounded channels with a reasonable capacity (e.g., 10000). When full, apply backpressure: drop oldest messages, use bounded_channel(n) with error handling, or yield if send would block.

#### F065 · 🟡 medium · [correctness] Incorrect decode reserve allows sequence budget bypass · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/domain/policy/continuous_batching.rs:256-257`
- **Verified:** partial (severity critical→medium)
- **Resolution:** ⊘ False alarm: `max_seqs` is enforced separately (`seqs_used >= seq_budget`). `decode_reserve` only governs KV budget, and a max_tokens=1 request genuinely needs 0 decode slots (its token comes from prefill).

**Problem.** The decode_reserve_for_new function returns max_tokens.saturating_sub(1), which produces 0 for max_tokens <= 1. This means a new sequence with max_tokens=1 reserves no decode capacity, allowing one extra sequence to be admitted beyond max_seqs. The admission control intended to reserve decode tokens for future generation is completely defeated for single-token sequences.

```rust
fn decode_reserve_for_new(max_tokens: usize) -> usize {
    max_tokens.saturating_sub(1)
}
```

**Fix.** Change to: fn decode_reserve_for_new(max_tokens: usize) -> usize { max_tokens.saturating_sub(1).max(1) } or simply return max_tokens, since every sequence needs at least 1 token for generation.

> **Verifier note:** The decode_reserve_for_new function returns max_tokens.saturating_sub(1), which produces 0 for max_tokens=1. This means new sequences with max_tokens=1 reserve no decode capacity, allowing them to consume full available KV budget during prefill without accounting for their mandatory decode phase. This can lead to decode requests being starved of budget later. However, the claim that this allows sequences to bypass max_seqs limits is incorrect - the sequence budget is enforced independently. Additionally, the practical impact is limited since max_tokens=1 is primarily used for diffusion requests, which are routed to a separate DiffusionPolicy rather than ContinuousBatchingPolicy.

#### F071 · 🟡 medium · [soundness] Potential integer overflow in RequestId allocator on skip-zero · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/infrastructure/transport/control_plane/pending_calls.rs:205-216`
- **Resolution:** ⊘ Requires 2^64 requests to hit; the skip-zero double-increment is correct for any realistic run. Not worth guarding.

**Problem.** The alloc_id() function attempts to skip RequestId(0) by fetching-adding twice if raw==0. However, if a u64 overflow causes the second fetch_add to also return 0 (astronomically unlikely but theoretically possible after 2^64 requests), the function returns RequestId(0), violating the documented invariant that 0 is reserved for uncorrelated traffic. This breaks the protocol assumption that RequestId(0) means no correlation.

```rust
let raw = self.next_id.fetch_add(1, Ordering::Relaxed);
let raw = if raw == 0 {
    self.next_id.fetch_add(1, Ordering::Relaxed) // Could also be 0
} else { raw };
```

**Fix.** Use a loop: loop { let raw = self.next_id.fetch_add(1, Ordering::Relaxed); if raw != 0 { return RequestId(raw); } }

#### F072 · ⚪ low · [correctness] Silent KV capacity loss from integer division truncation

- **Location:** `crates/infer-scheduler/src/config.rs:109-112`

**Problem.** apply_worker_capacity() computes num_gpu_blocks = tokens / block_size using integer division. If the worker reports tokens that are not a multiple of block_size, the remainder is silently truncated. For example, max_total_kv_tokens=1025 and block_size=256 gives 4 blocks (1024 tokens), losing 1 token of capacity. This is not a functional bug but represents silent KV pool capacity loss.

```rust
self.num_gpu_blocks = tokens / block_size;
```

**Fix.** Add validation: warn if tokens % block_size != 0, or round up with div_ceil and adjust num_gpu_blocks accordingly. Alternatively, validate that the worker always reports a multiple of block_size.


### infer-core · Tensor / storage / dtype / ports

> The core slice implements foundational abstractions (Tensor, Storage, device memory ports, exec scopes, and op traits) with generally solid structure and clear safety boundaries through Arc-based reference counting and typed slices. However, several high-severity correctness and soundness issues emerge:

Key concerns: (1) Tensor view construction (from_raw_parts, view_raw, narrow) lacks runtime validation of bounds and contiguity invariants, creating vectors for out-of-bounds access via unchecked pointer arithmetic. (2) Storage::alloc/Drop asymmetry on zero-byte tensors (size.max(1)) may corrupt device memory pools. (3) Dtype registry uses lock poisoning without recovery and relaxed atomics without synchronization, creating potential race conditions and DoS. (4) Reference kernels (scatter_kv_paged, attention_paged) make unsafe assumptions about plan/index tensor lengths without early validation. (5) Bitcast and pointer alignment are documented but not enforced at allocation time. The design is sound in principle—Arc ownership, explicit offset tracking, scope-based device context—but implementation gaps in validation layer create silent failures and subtle races that would manifest only at scale or under adversarial input.

#### F074 · 🟠 high · [correctness] Narrow may mark non-contiguous view as contiguous · ☑ FIXED

- **Location:** `crates/infer-core/src/tensor.rs:237`
- **Verified:** confirmed
- **Resolution:** ☑ `narrow` now keeps `dim==0` slices contiguous (a dim-0 sub-block stays contiguous).

**Problem.** The narrow function marks the result as contiguous only when start==0 && length==original_shape[dim]. However, after narrowing on dim=0 with start > 0, the view should still be contiguous (single linear block of memory) even though the offset has shifted. The condition is overly restrictive and will incorrectly mark valid contiguous narrow views as non-contiguous, causing unnecessary fallbacks.

```rust
let is_contig = self.is_contiguous && start == 0 && length == shape[dim];
```

**Fix.** Fix the is_contiguous logic for narrow: a view after narrow on dim is contiguous iff the parent was contiguous AND (dim == 0 OR length == shape[dim] OR dim is the innermost dimension with full extent). For the simple case on dim=0, the result is always contiguous if the parent was.

#### F073 · 🟠 high · [soundness] data_ptr arithmetic may overflow on large offsets · ☑ FIXED

- **Location:** `crates/infer-core/src/tensor.rs:281-286`
- **Verified:** confirmed
- **Resolution:** ☑ Added a debug-assert bounds check in `view_raw` (offset+numel ≤ storage bytes); validates once at view construction, hot-path `data_ptr` stays unchecked.

**Problem.** The data_ptr method computes `(self.storage.ptr() as *const T).add(self.offset_elems)` without verifying that the offset is within valid bounds. While the comment claims the caller of view_raw is responsible, there is no runtime validation in from_raw_parts or view_raw to ensure offset_elems + numel stays within storage.size().

```rust
#[inline]
    pub fn data_ptr(&self) -> *const T {
        // SAFETY: storage.ptr() is valid for storage.size() bytes; offset
        // stays within bounds by construction (caller of view_raw is
        // responsible).
        unsafe { (self.storage.ptr() as *const T).add(self.offset_elems) }
    }
```

**Fix.** Add a debug_assert! in from_raw_parts and view_raw to validate that (offset_elems + numel) * T::SIZE_BYTES <= storage.size(). This catches incorrect caller assumptions in development.

> **Verifier note:** The data_ptr method (lines 281-286) performs unsafe pointer arithmetic assuming offset_elems + numel fits within storage bounds. However, the public constructors from_raw_parts (lines 38-55) and view_raw (lines 172-188) do not validate this precondition. They accept arbitrary offset_elems and shape values from callers without checking (offset_elems + shape.numel()) * T::SIZE_BYTES <= storage.size(). While narrow and view_contiguous derive their parameters carefully, from_raw_parts and view_raw are public APIs that external code (or internal typos) could misuse to create tensors with out-of-bounds offset+size combinations. When such a tensor's data_ptr is used for memory operations (to_host_vec via from_raw_parts at line 154, as_slice at line 536, copy_from device operations), the unsafe pointer arithmetic proceeds unvalidated, risking memory safety violations.

#### F086 · 🟡 medium · [concurrency] DTypeId::register uses relaxed atomics without synchronization · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/dtype/mod.rs:26-27`
- **Resolution:** ⊘ Relaxed monotonic counter is correct for unique-id allocation; the RwLock write establishes happens-before for the spec insert. Startup-only path.

**Problem.** DTypeId::register uses AtomicU16 with Ordering::Relaxed to allocate IDs, then calls registry().write() to insert the spec. If two threads race to register, they could allocate different IDs (e.g., 1024 and 1025) but both insert into the registry. The relaxed ordering provides no synchronization between the atomic increment and the subsequent registry write, so on weakly ordered architectures (ARM), one thread's spec insertion could be reordered before its ID allocation, causing a use-after-register bug.

```rust
let id = DTypeId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
        registry()
            .write()
            .expect("dtype registry poisoned")
            .insert(id.0, spec);
```

**Fix.** Use Ordering::Release for fetch_add, or equivalently, acquire the registry write lock BEFORE allocating the ID to serialize registrations. This ensures the ID and spec are always synchronized.

#### F081 · 🟡 medium · [design] Dtype registry uses RwLock with poisoning but no recovery · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/dtype/mod.rs:28-31, 40-46, 53-59`
- **Resolution:** ⊘ Registry writes are startup-only (dtype registration); poisoning on a panicking registration is acceptable fail-fast, not a runtime DoS vector.

**Problem.** Multiple calls to registry().write().expect('dtype registry poisoned') will panic if a thread holding the write lock panics. This is a denial-of-service vector: a panicking type registration task will poison the global registry, making all subsequent DTypeId queries fail with a panic rather than a recoverable error.

```rust
registry()
            .write()
            .expect("dtype registry poisoned")
            .insert(id.0, spec);
```

**Fix.** Replace .expect() with proper error handling: e.g., return Err(OpError::Kernel(...)) if the lock is poisoned. Alternatively, use an IntMap or other lock-free structure (e.g., DashMap) to avoid lock poisoning altogether.

#### F088 · 🟡 medium · [soundness] Workspace pointer is not validated for alignment or size · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/exec.rs:130-136`
- **Resolution:** ⊘ `Workspace::from_raw` is an internal constructor fed by the backend's own arena; callers pass correctly-sized/aligned pointers by construction.

**Problem.** Workspace::from_raw accepts a NonNull<u8> and size without verifying that the size is correct or that the pointer is properly aligned for any ops that might use it. A caller could pass an undersized or misaligned pointer, leading to out-of-bounds writes or faults when kernels use the workspace.

```rust
pub fn from_raw(ptr: Option<NonNull<u8>>, size: usize) -> Self {
        Self {
            _ptr: ptr,
            _size: if ptr.is_some() { size } else { 0 },
            _d: PhantomData,
        }
    }
```

**Fix.** Document that Workspace::from_raw is unsafe and the caller must ensure size is correct and ptr is aligned to the largest alignment requirement of any ops using it. Consider adding debug_assert for size > 0 iff ptr.is_some().

#### F082 · 🟡 medium · [correctness] KvEdit::apply_step may incorrectly compute truncate range · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/kv.rs:167-172`
- **Resolution:** ⊘ False alarm: `accepted<=spec` and `current>=spec` are both checked, so `keep=current-spec+accepted<=current` (no overflow); `&mut self` means no concurrency.

**Problem.** In apply_step, when spec > 0 (speculative), the code computes keep = base + accepted_count. If accepted_count is very large, this could overflow. Additionally, the check current < spec on line 162 does not account for concurrent updates to seq_kv_len, leading to potential TOCTOU issues in a multi-threaded scenario.

```rust
let base = current - spec;
                let keep = base + accepted_count;
                if keep < current {
                    freed.extend(self.truncate(sid, keep)?);
                }
```

**Fix.** Use checked arithmetic for keep = base.checked_add(accepted_count) to prevent overflow. For TOCTOU: if seq_kv_len is modified concurrently by another thread, re-fetch current after the check or use atomic ops with CAS loops.

#### F084 · 🟡 medium · [correctness] scatter_kv_paged_reference may read out-of-bounds if block_tables is corrupted · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/ports/fused_ops.rs:391`
- **Verified:** partial (severity high→medium)
- **Resolution:** ⊘ Reference/CPU path only; the bounds check at fused_ops.rs:391 does gate the access. Prod runs the CUDA kernel.

**Problem.** block_for_position computes table_idx = batch_idx * plan.max_blocks_per_seq + block_slot, then checks if block_tables.get(table_idx) succeeds. However, if the plan is corrupted or block_tables is shorter than expected, the bounds check happens but the prior access at lines 378-382 already uses plan.batch and seq_lens_step without validating their correlation with the actual block_tables length.

```rust
let table_idx = batch_idx * plan.max_blocks_per_seq + block_slot;
    let Some(&block) = block_tables.get(table_idx) else {
        return Err(...)
```

**Fix.** Validate that block_tables.len() == plan.batch * plan.max_blocks_per_seq at the start of scatter_kv_paged_reference and attention_paged_reference. Pre-compute this invariant rather than relying on .get() to catch errors after partial computation.

#### F075 · 🟡 medium · [soundness] Double-free risk on storage due to missing size validation · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/storage.rs:38-44`
- **Resolution:** ⊘ `size.max(1)` is symmetric between alloc and free (Drop also uses `size.max(1)`), and the CUDA pool keys by the requested size — no asymmetry or double-free.

**Problem.** Storage::alloc calls device.alloc_bytes(size.max(1)), which may succeed and return a pointer for an empty tensor (size=0). Later, Drop calls device.free_bytes with size.max(1), creating asymmetry: the device allocated for a 1-byte dummy but the domain intended zero bytes. If the device's free pool tracks by exact size, the dummy won't recycle correctly and may leak or double-free if re-interpreted as a valid allocation.

```rust
pub fn alloc(device: &D, size: usize) -> OpResult<Arc<Self>> {
        let ptr = device.alloc_bytes(size.max(1))?;
        Ok(Arc::new(Self {
            ptr,
            size,
            device: device.clone(),
        }))
    }
```

**Fix.** Store the actual allocated size (size.max(1)) separately, or check at Drop that you free with the same size.max(1) used at alloc. The asymmetry between alloc(size) and free(size.max(1)) is a source of pool corruption.

#### F078 · 🟡 medium · [soundness] to_host_vec may succeed with empty tensor but violates Vec invariants · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/tensor.rs:160-164`
- **Resolution:** ⊘ `Vec::with_capacity(0)` + `set_len(0)` is well-defined (empty Vec, dangling-but-aligned ptr); no allocator-contract violation. Verifier-adjacent to the refuted claims.

**Problem.** When downloading a zero-byte tensor, the code creates an empty Vec, skips the download, then calls set_len(0). However, Vec::with_capacity(0) is valid and the subsequent set_len(0) is safe. The real issue is that after zero-byte download, the code does not initialize the Vec's internal pointer, and set_len(0) on a zero-capacity Vec is undefined behavior per Rust's allocator contract.

```rust
} else {
            // SAFETY: empty vec.
            unsafe {
                out.set_len(0);
            }
        }
```

**Fix.** For empty tensors, just return Vec::new() or Vec::with_capacity(0) without calling set_len. The safety comment is misleading; set_len requires the buffer to be initialized (even if empty).

#### F077 · 🟡 medium · [soundness] Unsafe assume about storage alignment in data_ptr_mut without validation · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/tensor.rs:295-298`
- **Resolution:** ⊘ Backends allocate with alignment ≥ the dtype's requirement by construction; `data_ptr_mut` mirrors `data_ptr`. Adding per-call alignment checks on the hot path isn't warranted.

**Problem.** data_ptr_mut returns a *mut T by casting storage.ptr() and adding offset_elems without validating that storage.ptr() is actually aligned for type T. If a backend allocates with weaker alignment than T requires, dereferencing the result is UB.

```rust
#[inline]
    pub fn data_ptr_mut(&self) -> *mut T {
        // SAFETY: see data_ptr.
        unsafe { (self.storage.ptr() as *mut T).add(self.offset_elems) }
    }
```

**Fix.** Document that Storage must be allocated with alignment suitable for the strongest dtype that will be stored. Add a compile-time or runtime check in backends' alloc_bytes to enforce minimum alignment (e.g., 16 bytes for SIMD ops on both CPU and CUDA).

#### F080 · 🟡 medium · [correctness] copy_from may copy overlapping memory even after checking storage pointers · ⊘ WONT-FIX

- **Location:** `crates/infer-core/src/tensor.rs:366`
- **Resolution:** ⊘ `copy_from` compares `Arc::as_ptr` for the common aliasing case; cross-device overlap of distinct allocations isn't a real scenario in this engine (one device per worker).

**Problem.** copy_from checks if self.storage and src.storage are the same Arc by comparing pointers with Arc::as_ptr. However, two distinct Arc instances could point to the same underlying allocation (e.g., if one was cloned from the other then one Arc was dropped—this is rare but possible if the Arc refcount is not stable). More critically, even if the Arc pointers are different, the views could still overlap if they alias different parts of separate device buffers on different logical devices.

```rust
if Arc::as_ptr(&self.storage) == Arc::as_ptr(&src.storage) {
```

**Fix.** The Arc pointer check is actually correct for detecting shared ownership. However, add a clarifying comment that device D2D copies require non-overlapping regions, and the caller is responsible for ensuring the views don't alias if using different storage arcs. Consider adding a check that validates no overlap even with different storages (comparing memory ranges), but this is complex for device memory.

#### F089 · ⚪ low · [design] Fp8 dtypes have incorrect StorageDtype base

- **Location:** `crates/infer-core/src/dtype/mod.rs:166-169, 181-184`

**Problem.** Fp8E4m3 and Fp8E5m2 are declared with StorageDtype base having DATA_TYPE = DataType::I8. This is technically correct (they are 1-byte values), but semantically confusing: DataType::I8 is for signed 8-bit integers, not floating-point. The mismatch doesn't break anything, but it muddies the type semantics.

```rust
impl StorageDtype for Fp8E4m3 {
    const DATA_TYPE: DataType = DataType::I8;
    const SIZE_BYTES: usize = 1;
}
```

**Fix.** Consider adding a DataType::F8 variant (or F8E4M3, F8E5M2), or document that F8 types reuse I8 as a storage placeholder. This clarifies intent without changing runtime behavior.

#### F087 · ⚪ low · [design] ActiveGuard does not enforce single-device context invariant

- **Location:** `crates/infer-core/src/exec.rs:89-108`

**Problem.** ActiveGuard uses _not_send: PhantomData<Rc<()>> to prevent Send/Sync, ensuring it stays on one thread. However, the _prev_device restoration in Drop is not conditional: it always calls restore_device, even if enter_device/restore_device are no-ops. For CUDA, this is wasteful; for CPU, it is unnecessary. The pattern works but is not elegantly documented.

```rust
pub struct ActiveGuard<'a, D: ExecDevice> {
    _scope: &'a D::Scope,
    _prev_device: DeviceId,
    _not_send: PhantomData<Rc<()>>,
}
```

**Fix.** Add a doc comment explaining that ActiveGuard assumes the device implements enter_device/restore_device pair semantics, and that repeated guards on the same device may incur redundant context switches. Consider making this a true guard only for multi-GPU setups.

#### F079 · ⚪ low · [design] Contiguous strides computed but not validated in from_host_slice

- **Location:** `crates/infer-core/src/tensor.rs:96`

**Problem.** from_host_slice always uses contiguous_strides() without allowing the caller to specify strides. This is correct for the intended use case (loading from dense host buffers), but the API does not clearly document that non-contiguous layouts are unsupported on load.

```rust
let strides = shape.contiguous_strides();
```

**Fix.** Add a doc comment to from_host_slice clarifying that only contiguous row-major layouts are supported. If non-contiguous load is needed, use from_raw_parts with custom strides (though no public API currently provides this).


### CUDA backend · Core + hot kernels (GEMM/SDPA, flash attn, KV, sampler)

#### F094 · 🟡 medium · [concurrency] Arena overflow rollback is not atomic (TOCTOU race) · ☑ FIXED

- **Location:** `crates/infer-backend-cuda/src/config.rs:243-246`
- **Resolution:** ☑ Arena `arena_alloc` now reserves via `fetch_update` (never transiently over-commits) instead of fetch_add+rollback.

**Problem.** The arena allocation pattern uses fetch_add then post-checks if it overflowed, then conditionally fetch_sub. This is not atomic and can race: between fetch_add and the if-check, another thread could increment arena_off further. If the check finds overflow and tries to rollback via fetch_sub, it only subtracts its own allocation, not reverting the intervening additions. On a multi-threaded scheduler with concurrent requests allocating from the arena simultaneously, this can cause arena_off to exceed GRAPH_ARENA_SIZE and subsequent allocations to return pointers beyond the arena boundary, aliasing with other memory.

```rust
Lines 243-246: `let off = self.arena_off.fetch_add(n, Ordering::AcqRel); if off + n > GRAPH_ARENA_SIZE { self.arena_off.fetch_sub(n, Ordering::AcqRel); return None; }`. Between fetch_add (line 243) and fetch_sub (line 245), another thread's allocation can increment arena_off.
```

**Fix.** Use compare-and-swap (compare_exchange) in a loop to atomically reserve space, or protect the arena_off with a Mutex during multi-threaded access. Example: `loop { let off = self.arena_off.load(Ordering::Acquire); if off + n > GRAPH_ARENA_SIZE { return None; } if self.arena_off.compare_exchange(off, off + n, Ordering::Release, Ordering::Acquire).is_ok() { return allocated_ptr; } }`. Alternatively, document that arena_alloc is single-threaded (per-stream) and add a comment explaining the assumed execution model.

#### F092 · 🟡 medium · [performance] Per-token allocation churn in eager SDPA (non-graph-captured decode) · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cuda/src/kernels/matmul/sdpa.rs:187-198, 259, 296`
- **Resolution:** ⊘ Only the non-graph-captured eager fallback allocates; the primary decode path uses the capture arena (already zero-alloc). Eager is a correctness fallback, not the hot path.

**Problem.** The sdpa() function allocates 7 intermediate GPU tensors (q_hsd, k_hsd_kv, v_hsd_kv, scores, attn, out_hsd, v_hds) on every call. During eager (non-captured) decode, this happens per-token per-layer, causing allocation churn even though the shapes are deterministic and reusable. Graph capture mitigates this via the arena, but eager forward still pays the cost. The code comment (lib.rs:682-688) acknowledges 'every layer of every step used to allocate... (~18µs × num_layers / token TTOT)', but sdpa itself still allocates.

```rust
Lines 187, 197-198: `let q_hsd: Tensor<T, Cuda> = Tensor::zeros([num_heads, seq, head_dim], &dev)?;` and similar for k_hsd_kv, v_hsd_kv (lines 197-198). Lines 259, 296, 306, 314 allocate scores, attn, out_hsd, v_hds. These happen unconditionally every call.
```

**Fix.** For the non-captured eager path, pass pre-allocated scratch tensors to sdpa() instead of allocating internally, similar to the caller-allocated workspace pattern used in flash_attn_gqa::attention_paged (lib.rs lines 706-730). This requires an API change to accept optional scratch buffers.

#### F093 · 🟡 medium · [correctness] Missing workspace size validation in release builds · ☑ FIXED

- **Location:** `crates/infer-backend-cuda/src/lib.rs:706-715, 513-527`
- **Resolution:** ☑ Flash-decode workspace size check is now ALWAYS on (returns OpError) instead of `#[cfg(debug_assertions)]` — prevents silent OOB device writes in release.

**Problem.** The decode path allocates a workspace buffer sized via flash_decode_workspace_capacity_f32(), but the validation that it's large enough (lines 513-527) is guarded by #[cfg(debug_assertions)], meaning it's skipped in release builds. If the caller allocates a buffer too small for the actual batch size (e.g., due to a planning error), the flash kernel will write out-of-bounds, causing silent memory corruption or crashes. The workspace capacity depends on batch and head_dim, which can vary between capture and replay.

```rust
Lines 513-527 wrapped in #[cfg(debug_assertions)] check `flash_attn_batched_decode_workspace_bytes()` vs actual allocation. This entire block is removed in release. Lines 706-715 and 734-739 show the workspace is sized once and reused; no runtime check in release validates it's adequate for the current batch.
```

**Fix.** Add a release-mode check (not just debug) that verifies workspace.numel() >= flash_decode_workspace_capacity_f32(batch, head_num, head_dim). Either return OpError::Kernel or panic!() if undersized, to catch configuration errors early rather than corrupt memory silently.

#### F096 · ⚪ low · [performance] Inefficient arena memset on every allocation

- **Location:** `crates/infer-backend-cuda/src/config.rs:251-253`

**Problem.** Every arena allocation calls cudaMemsetAsync to zero-initialize (lines 251-253). While async, this still enqueues a device command per allocation. During graph-captured decode with ~190 transient allocations per step, this results in 190+ memset commands per step. A more efficient approach is to pre-zero the entire arena once at capture_begin and reuse zero-initialized regions via narrow/slice, or use a fast memory pool for pre-zeroed blocks.

```rust
Lines 251-253: `unsafe { ffi::cudaMemsetAsync(ptr, 0, n, self.stream); }` is called inside arena_alloc for every allocation. Line 107 comment mentions ~190 transient scratch tensors per step during decode.
```

**Fix.** Pre-zero the entire arena once at graph_capture_begin or allocate it with cudaMallocHost followed by cudaMemset before capture. Alternatively, use a simple allocation counter + marking scheme to track which regions are actively in use, and only memset on reallocation. For the common case of Tensor::zeros, verify that the narrow/slice pattern already relies on prior memset and does not re-zero.

#### F095 · ⚪ low · [correctness] Hardcoded GEMM workspace may be insufficient for cuDNN or future kernels

- **Location:** `crates/infer-backend-cuda/src/config.rs:9, 140-142`

**Problem.** The global GEMM workspace is hardcoded to 4 GiB (line 9). This size is passed to both cuBLASLt matmuls and cuDNN attention operations. If future CUDA/cuDNN versions require more workspace or if a different backend is used, the kernel will silently overflow the workspace buffer. The workspace is validated nowhere at config creation time; it's simply allocated and reused. A backend update could cause memory corruption.

```rust
Line 9: `const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 4usize * 1024 * 1024 * 1024;` is hardcoded. Lines 140-142 allocate this once. Lines 395-396 (flash_attn_gqa) and 157-159 (matmul) pass this workspace size to kernels without validation.
```

**Fix.** Query the actual workspace requirements from cuBLASLt and cuDNN at initialization (e.g., cublasLtMatmulAlgoGetHeuristic, cudnnGetConvolutionForwardWorkspaceSize) and allocate the maximum. If queries fail or return > 4GiB, either allocate more or error at init time. Document the workspace sizing assumptions.


### CUDA backend · Elementwise / fused / norm / rope kernels

> The CUDA kernel wrappers in this slice demonstrate solid type-safe abstractions over low-level C kernels, but have several critical correctness issues centered on unchecked arithmetic (usize→i32 casts without bounds checking), unchecked shape/stride validation, and potential integer overflow in byte offset calculations. Notably, qkv_norm_rope_scatter contains a documented but incompletely fixed issue regarding out-of-bounds reads when positions arrays are capacity-allocated but only prefix-filled. The softmax, broadcast_mul, groupnorm, and embedding kernels all compute derived dimensions (rows, spatial, etc.) without verifying divisibility or bounds. There are also performance concerns in the swiglu_packed F32 fallback which allocates tensors per-token, and design issues in layernorm where unsupported dtypes silently skip operations.

#### F100 · 🟡 medium · [correctness] Unchecked division in broadcast kernels · ☑ FIXED

- **Location:** `crates/infer-backend-cuda/src/kernels/broadcast_mul/mod.rs:65, 110`
- **Verified:** partial (severity high→medium)
- **Resolution:** ☑ `broadcast_mul_inplace`/`broadcast_add_inplace` now validate `x.numel() % dim == 0` and `dim != 0` before the division.

**Problem.** Lines 65, 110 in broadcast_mul and line 98 in broadcast_add compute row counts via unchecked division: `rows = x.numel() / dim` without verifying that the result is correct or that dim is a divisor of x.numel(). This can lead to incorrect kernel parameters.

```rust
// broadcast_mul/mod.rs line 65
let rows = (x.numel() as i32) / dim;
```

**Fix.** Validate divisibility: `if x.numel() % dim != 0 { return Err(OpError::Shape(format!("broadcast_mul: {} numel not divisible by {}", x.numel(), dim))); }`

#### F105 · 🟡 medium · [soundness] Potential integer overflow in byte size calculations in pad.rs · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:106, 172`
- **Resolution:** ⊘ Byte-size products are bounded by tensor capacity (allocated from VRAM, far below usize::MAX). Cannot overflow in practice.

**Problem.** Lines 106, 172 in pad.rs compute `bytes_per_row = d * T::SIZE_BYTES` and then `src_bytes = n * bytes_per_row`. These are usize multiplications. If n or d are very large, the product can overflow, silently wrapping. The overflowed size is then passed to cudaMemcpyAsync, causing a memcpy of the wrong size.

```rust
let bytes_per_row = d * T::SIZE_BYTES;
let src_bytes = n * bytes_per_row;  // potential overflow
unsafe {
    d2d(
        dst.data_ptr_mut() as _,
        src.data_ptr() as _,
        src_bytes,  // <-- could be wrong due to overflow
        stream,
    )?
}
```

**Fix.** Use checked arithmetic: `let bytes_per_row = d.checked_mul(T::SIZE_BYTES).ok_or(OpError::Shape(...))?; let src_bytes = n.checked_mul(bytes_per_row).ok_or(OpError::Shape(...))?;`

#### F098 · 🟡 medium · [correctness] Unchecked usize to i32 casting can cause integer overflow · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:122, 129`
- **Verified:** partial (severity high→medium)
- **Resolution:** ⊘ Casts are bounded by model dims (hidden ≤ ~16k, tokens ≤ max_batch_tokens=8192), far below i32::MAX; cannot overflow with real weights. Guarding dozens of launch wrappers is churn.

**Problem.** Unchecked casting of usize to i32 throughout the kernel launchers can cause silent integer overflow. Lines 122, 129 in pad.rs cast pad_rows and d to i32; line 72 in groupnorm casts spatial product; lines 46-48 in embedding cast vocab/dim/seq_len; line 65 in broadcast_mul casts rows. On 64-bit systems, if any of these values exceed 2^31-1, the cast silently truncates, causing the kernel to process the wrong dimensions.

```rust
// pad.rs line 122
broadcast_row_bf16_forward(
    dst_pad_base as *mut bf16,
    pad_token.data_ptr() as *const bf16,
    pad_rows as i32,  // <-- unchecked cast
    d as i32,         // <-- unchecked cast
    stream,
),

// groupnorm/mod.rs line 72
let spatial: i32 = shape[2..].iter().product::<usize>() as i32;
```

**Fix.** Replace all unchecked casts with checked conversion: `let pad_rows_i32 = i32::try_from(pad_rows).map_err(|_| OpError::Shape("dimension overflow"))?;`. Apply this systematically to pad.rs, groupnorm, embedding, broadcast_mul, and other kernel launchers.

> **Verifier note:** Unchecked usize-to-i32 casts exist in kernel launchers (pad.rs lines 122-129, groupnorm/mod.rs line 72, embedding/mod.rs lines 46-48, broadcast_mul/mod.rs lines 64-65) and could theoretically cause integer overflow. However, practical LLM inference constraints make this unlikely: default max_model_len=4096, typical hidden_size<12k, and GPU memory limits prevent allocating tensors that would overflow. Tensors originate from validated allocations, not untrusted input. The issue is a correctness risk (should use checked conversion per best practices) but not a realistic vulnerability in deployed systems. Suggested fix (try_from with error handling) is valid but addresses a low-probability scenario.

#### F103 · 🟡 medium · [correctness] No validation that device positions pointer is valid in rope kernels · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cuda/src/kernels/rope/mod.rs:60-71`
- **Resolution:** ⊘ `positions_dev` is produced by the worker's own upload path with a known length; it's an internal contract, not external input.

**Problem.** The rope_inplace function accepts a raw device pointer `positions_dev: *const i32` and a count `num_tokens: i32`, but provides no validation that the device array is valid or contains num_tokens elements. If the pointer is NULL, invalid, or points to an array smaller than num_tokens, the kernel will access out-of-bounds memory.

```rust
pub fn rope_inplace<T: Dtype>(
    stream: cudaStream_t,
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions_dev: *const i32,  // <-- raw pointer, no validation
    num_tokens: i32,
    head_num: i32,
    kv_head_num: i32,
    head_dim: i32,
) -> OpResult<()>
```

**Fix.** Either accept `&Tensor<i32, Cuda>` for positions instead of a raw pointer (type-safe), or document a precondition that positions_dev must point to at least num_tokens i32 values on the device. Consider adding a debug-only assertion.

#### F106 · 🟡 medium · [performance] Per-token memory allocation in swiglu_packed F32 fallback · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cuda/src/kernels/swiglu/mod.rs:110-111`
- **Resolution:** ⊘ Only the F32 fallback allocates; production runs BF16. Not the hot path.

**Problem.** Lines 110-111 in swiglu/mod.rs allocate two new Tensor objects (gate and up) on every F32 fallback call. If swiglu_packed is called in the decode hot path (per-token), this results in per-token GPU memory allocations via Tensor::zeros(), which is inefficient and can cause memory fragmentation.

```rust
DataType::F32 => {
    let dev = gate_up.device().clone();
    let mut gate: Tensor<T, Cuda> = Tensor::zeros([rows, inter], &dev)?;  // allocation
    let mut up: Tensor<T, Cuda> = Tensor::zeros([rows, inter], &dev)?;    // allocation
```

**Fix.** Either implement a dedicated swiglu_packed_fp32 CUDA kernel, or refactor to accept pre-allocated workspace tensors from the caller (similar to conv2d workspace pattern).

#### F107 · ⚪ low · [design] Silent no-op on unsupported dtype in layernorm

- **Location:** `crates/infer-backend-cuda/src/kernels/layernorm/mod.rs:157, 183`

**Problem.** Lines 157 and 183 in layernorm have `_ => {}` branches that silently do nothing if an unsupported dtype is encountered. This means the weight/bias application could be completely skipped without error, leading to incorrect layernorm output.

```rust
match T::DATA_TYPE {
    DataType::F32 => broadcast_mul_f32_forward(...),
    DataType::BF16 => broadcast_mul_bf16_forward(...),
    DataType::F16 => broadcast_mul_f16_forward(...),
    _ => {}  // Silent no-op!
}
```

**Fix.** Replace `_ => {}` with explicit error: `_ => return Err(OpError::Kernel(format!("layernorm: unsupported dtype {:?}", T::DATA_TYPE)))`

#### F097 · ⚪ low · [correctness] No stride validation in rmsnorm for non-contiguous tensors

- **Location:** `crates/infer-backend-cuda/src/kernels/rmsnorm/mod.rs:139-157`
- **Verified:** partial (severity high→low)

**Problem.** The rmsnorm derive_layout function handles 2D and 3D tensors with non-contiguous strides, but it doesn't validate that the strides are sensible. For a 3D tensor, it directly uses strides[0] and strides[1] without checking if they allow the kernel to access valid data. A malformed stride could cause the CUDA kernel to read past the buffer bounds.

```rust
match t.ndim() {
    2 => Ok(Layout {
        outer0: shape[0] as i32,
        outer1: 1,
        dim: dim as i32,
        stride0: strides[0] as i64,
        stride1: 0,
    }),
    3 => Ok(Layout {
        outer0: shape[0] as i32,
        outer1: shape[1] as i32,
        dim: dim as i32,
        stride0: strides[0] as i64,
        stride1: strides[1] as i64,
    }),
```

**Fix.** Validate that strides[0] >= dim and strides[1] >= dim (for 3D case). For non-contiguous layouts, also validate that the maximum offset computed as (outer0-1)*stride0 + (outer1-1)*stride1 + (dim-1) does not exceed the tensor's numel().

> **Verifier note:** The derive_layout function validates that the last stride equals 1 (line 124), ensuring the innermost dimension is contiguous. While additional validation of stride0 and stride1 (e.g., alignment requirements, or bounds checks) would provide defense-in-depth, the current code is not demonstrably unsafe in practice because: (1) all production calling patterns create tensors with valid strides through safe APIs, and (2) misuse would require intentional circumvention of the documented `view_raw` caller-responsibility contract. The CUDA kernel itself has undocumented alignment requirements (stride0/1 must be 8-byte aligned for bf16) that aren't validated, but these are kernel implementation details, not correctness bugs in the Rust code.

#### F104 · ⚪ low · [correctness] Deferred shape validation in rope_interleaved

- **Location:** `crates/infer-backend-cuda/src/kernels/rope_interleaved/mod.rs:67-74`

**Problem.** The rope_interleaved kernel does not validate that cos and sin have matching shapes or that they are [seq, head_dim/2]. If they are mismatched or missing dimensions, the assertion on lines 69 will fail, but this happens after the dtype dispatch, not before. This could lead to a panicked/error state after unsafe operations have begun.

```rust
let cs = cos.shape().as_slice();
let ss = sin.shape().as_slice();
if cs != [seq, half] || ss != [seq, half] {
    return Err(OpError::Shape(...));
}
```

**Fix.** Move shape validation to the start of the function, before any tensor operations or dtype dispatch.


### CPU backend (reference / fallback)

#### F108 · 🟠 high · [correctness] Integer overflow in ewise_mul lacks input bounds validation · ☑ FIXED

- **Location:** `crates/infer-backend-cpu/src/lib.rs:181-194`
- **Verified:** confirmed
- **Resolution:** ☑ `ewise_mul` now calls `check_contiguous3`/`check_numel3` like the other elementwise ops.

**Problem.** ewise_mul does not validate that a, b, and dst have equal numel(). Unlike add() and add_inplace(), which call check_numel3(), this operation could read/write beyond bounds if shapes mismatch. When i reaches a.numel(), the loop continues accessing b and dst at indices that may be out-of-bounds.

```rust
fn ewise_mul<T: Dtype>(
    a: &Tensor<T, Self>,
    b: &Tensor<T, Self>,
    dst: &mut Tensor<T, Self>,
) -> OpResult<()> {
    for i in 0..a.numel() {
        // ... no check that b.numel() == a.numel() or dst.numel() == a.numel()
    }
}```

**Fix.** Add `check_numel3(a, b, dst)?;` at the start of ewise_mul, matching the pattern in add().

#### F110 · 🟠 high · [correctness] Potential out-of-bounds access in embedding operation · ☑ FIXED

- **Location:** `crates/infer-backend-cpu/src/lib.rs:290-309`
- **Verified:** confirmed
- **Resolution:** ☑ `embedding` now validates every index against `[0, vocab)` before the raw copy.

**Problem.** embedding() does not validate that each index in the indices tensor is within [0, table.shape()[0]). If idx >= table.numel()/dim, the pointer arithmetic idx * dim * T::SIZE_BYTES will access beyond the table allocation. This is a classic bounds violation on external input (indices).

```rust
let idx_slice = unsafe { std::slice::from_raw_parts(indices.data_ptr(), seq_len) };
for i in 0..seq_len {
    let idx = idx_slice[i] as usize;  // idx is not validated
    unsafe {
        std::ptr::copy_nonoverlapping(
            (table.data_ptr() as *const u8).add(idx * dim * T::SIZE_BYTES),  // OOB if idx too large
            (output.data_ptr_mut() as *mut u8).add(i * dim * T::SIZE_BYTES),
            dim * T::SIZE_BYTES,
        );
    }
}```

**Fix.** Add bounds check: before the copy, verify `idx < table.shape().as_slice()[0]`. Return OpError::Shape if violated.

> **Verifier note:** The embedding() function in crates/infer-backend-cpu/src/lib.rs (lines 290-309) does not validate that each index in the indices tensor is within [0, table.shape()[0]). When casting idx from i32 to usize at line 299, if idx >= table.shape()[0], the pointer arithmetic idx * dim * T::SIZE_BYTES at line 302 will compute an offset beyond the table allocation, causing the copy_nonoverlapping to access out-of-bounds memory. This contrasts with the CUDA implementation which explicitly checks token bounds before access. The severity is high because this is a classic use-after-free/OOB vulnerability on untrusted input (embedding indices).

#### F109 · 🟠 high · [soundness] Unsafe pointer arithmetic without bounds checks in split_cols · ☑ FIXED

- **Location:** `crates/infer-backend-cpu/src/lib.rs:357-385`
- **Verified:** partial
- **Resolution:** ☑ `split_cols` now checks contiguity and that `rows*total_cols`/`rows*dst_cols` fit the backing tensors.

**Problem.** split_cols checks col_offset + dst_cols <= total_cols but does not verify that the source tensor is actually [rows, total_cols] or that dst is [rows, dst_cols]. The pointer arithmetic r * total_cols + col_offset could overflow or access invalid memory if rows/total_cols don't match the actual tensor layout. Missing is_contiguous() checks (present in other ops like add).

```rust
fn split_cols<T: Dtype>(
    src: &Tensor<T, Self>,
    dst: &mut Tensor<T, Self>,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
) -> OpResult<()> {
    if col_offset + dst_cols > total_cols {
        return Err(...);
    }
    // But no check that src.shape() == [rows, total_cols], or that they're contiguous
    let src_ptr = src.data_ptr();
    let dst_ptr = dst.data_ptr_mut();
    for r in 0..rows {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(r * total_cols + col_offset),  // Could overflow
                dst_ptr.add(r * dst_cols),
                dst_cols,
            );
        }
    }```

**Fix.** Add shape and contiguity validation: check src.shape().as_slice() == &[rows, total_cols], dst.shape().as_slice() == &[rows, dst_cols], and that both are contiguous. Perform safe-checked multiplication for r * total_cols to prevent overflow.

> **Verifier note:** split_cols lacks validation that tensor shapes match the provided parameters (src should be [rows, total_cols], dst should be [rows, dst_cols]) and does not verify contiguity, despite doing unsafe pointer arithmetic. While this is currently mitigated by callers always providing correct shapes (shapes are baked into tensor allocation), the absence of defensive checks creates a soundness gap, especially in a hot path. This violates the pattern established by add() and other ops which validate via check_contiguous3() before similar arithmetic.

#### F112 · 🟡 medium · [correctness] Missing contiguity checks in matmul_quant · ☑ FIXED

- **Location:** `crates/infer-backend-cpu/src/lib.rs:219-246`
- **Resolution:** ☑ `matmul_quant` now checks contiguity, `group_size` divisibility, and operand sizes vs declared m/n/k.

**Problem.** matmul_quant lacks check_contiguous3() validation (present in add, add_inplace, etc.), and also does not validate scales shape. Pointer arithmetic assumes contiguous memory; non-contiguous tensors could cause incorrect computation or OOB access.

```rust
fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
    input: &Tensor<A, Self>,
    weight: &Tensor<W, Self>,
    output: &mut Tensor<O, Self>,
    scales: &Tensor<A, Self>,
    _zeros: Option<&Tensor<W, Self>>,
    group_size: usize,
) -> OpResult<()> {
    // No contiguity or shape checks; pointer arithmetic assumes row-major layout
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let group = p / group_size;
                let scale = unsafe { read_f64(scales.data_ptr().add(j * (k / group_size) + group)) };```

**Fix.** Add validation: check that input, weight, output, and scales are all contiguous. Also validate scales.shape() == [n, k/group_size] or equiv.

#### F113 · 🟡 medium · [soundness] Unsafe cast from *const T to *const u8 loses alignment information in read_f64 · ⊘ WONT-FIX

- **Location:** `crates/infer-backend-cpu/src/lib.rs:873-882`
- **Resolution:** ⊘ Sound: casting to `*const u8` gives alignment-1, and reading bytes has no alignment requirement regardless of T. No UB.

**Problem.** read_f64 casts *const T to *const u8 and creates a slice from it. If T has stricter alignment than u8 (e.g., f32 on some platforms), the cast could create a misaligned pointer. The slice is then reinterpreted as bytes, which is safe, but if the original pointer was misaligned for T, this is unsound.

```rust
unsafe fn read_f64<T: Dtype>(ptr: *const T) -> f64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, T::SIZE_BYTES) };
    // ... uses bytes
}```

**Fix.** Cast to *const u8 first, then ensure alignment is preserved: use `ptr as *const u8` or `ptr.cast::<u8>()`. If T requires alignment, verify ptr satisfies alignment before casting.

#### F114 · ⚪ low · [design] Integer division without validation in split_cols

- **Location:** `crates/infer-backend-cpu/src/lib.rs:357-385`

**Problem.** split_cols takes rows and total_cols as separate parameters, but doesn't verify they match src.shape(). If caller passes wrong values, pointer arithmetic will silently compute wrong indices. Compare with other ops that derive rows/cols from tensor shapes directly.

```rust
fn split_cols<T: Dtype>(
    src: &Tensor<T, Self>,
    dst: &mut Tensor<T, Self>,
    rows: usize,
    total_cols: usize,
    col_offset: usize,
    dst_cols: usize,
) -> OpResult<()> {
    // rows and total_cols are passed in, not validated against src.shape()```

**Fix.** Replace explicit rows/total_cols parameters with extraction from src.shape(). Validate len(src.shape()) == 2.

#### F111 · ⚪ low · [performance] Data race on SDPA scores allocation in hot loop

- **Location:** `crates/infer-backend-cpu/src/lib.rs:745-814`
- **Verified:** partial (severity high→low)

**Problem.** sdpa allocates a new Vec<f64> for scores on every token and every head: `let mut scores = vec![0.0f64; seq_len];` at line 762 inside nested loops (t and h). For a model with seq_len=4096 and num_heads=32, this is 32*4096=131k allocations per batch. This is a per-token / per-layer hot-path allocation that could be hoisted to a preallocated buffer.

```rust
for h in 0..num_heads {
    let kv_h = h / kv_mul;
    for t in 0..seq_len {
        let mut scores = vec![0.0f64; seq_len];  // ALLOCATES EVERY ITERATION
        for s in 0..seq_len {
            // ... compute softmax ...
        }
    }
}```

**Fix.** Allocate scores buffer once outside the outer loops: `let mut scores = vec![0.0f64; seq_len];` before `for h in 0..num_heads`, and clear it in each iteration of the t loop.

> **Verifier note:** CPU backend SDPA allocates Vec<f64> on every (head, token) iteration instead of once per head. Lines 762 allocates num_heads * seq_len times per call. However, this is a test/reference implementation (diffusion runs on CUDA only per project policy) and not a production hot path. While the allocation pattern is suboptimal, the issue is not a data race and the severity is inappropriately high for code that doesn't execute in production.


### Server · OpenAI HTTP API + ZMQ client bridge

#### F119 · 🟡 medium · [design] No timeout on parallel image generation futures · ⊘ WONT-FIX

- **Location:** `crates/infer-server/src/api/openai/images.rs:121`
- **Resolution:** ⊘ Already bounded: `infer()` registers a per-request deadline enforced by the ZMQ thread's `cancel_timed_out_requests`, so a hung image request is capped at `request_timeout_secs`.

**Problem.** `try_join_all()` on diffusion image futures has no individual timeout or deadline. If any single image generation hangs (e.g., scheduler doesn't respond), the entire HTTP request hangs indefinitely, blocking the worker thread and potentially consuming all worker slots if requests pile up.

```rust
let encoded_images = try_join_all(image_futures).await?;
```

**Fix.** Wrap image_futures with `tokio::time::timeout()` to apply a timeout to the entire join_all operation, matching the client request timeout.

#### F117 · 🟡 medium · [design] Per-token JSON serialization failure swallows error detail · ☑ FIXED

- **Location:** `crates/infer-server/src/api/openai/streaming.rs:19-30`
- **Resolution:** ☑ SSE serialization failure now emits a distinguishable `error` event instead of a fake `[DONE]` terminator.

**Problem.** When `serde_json::to_string()` fails in `json_event()`, the code logs an error and then yields `[DONE]` instead of an error chunk. This loses the actual serialization error from the client view and silently terminates the stream, making debugging impossible. The client sees only `[DONE]` with no indication of failure.

```rust
fn json_event<T: serde::Serialize>(request_id: &str, payload: &T) -> Event {
    match serde_json::to_string(payload) {
        Ok(data) => Event::default().data(data),
        Err(error) => {
            tracing::error!(...)
            Event::default().data("[DONE]")
        }
    }
}
```

**Fix.** Return a Result type or yield a proper error chunk (with finish_reason: "error") instead of `[DONE]`, so the client knows the stream failed.

#### F122 · 🟡 medium · [performance] Mutex contention on Waker pipe per command submit · ⊘ WONT-FIX

- **Location:** `crates/infer-server/src/client/zmq_client.rs:108-115`
- **Resolution:** ⊘ Mutex is held only for a 1-byte pipe write, never across await; effectively uncontended. A prior inproc-PAIR waker was removed deliberately (see comment). Not a real bottleneck.

**Problem.** Every request submit (infer or infer_stream) calls `self.waker.wake()`, which acquires a Mutex to write 1 byte to the pipe. Under high concurrency (many axum worker threads submitting requests), this mutex becomes a bottleneck. The lock is held across the write, and a contended lock can add microseconds of latency per submit.

```rust
fn wake(&self) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.write(&[1u8]);
        }
    }
```

**Fix.** Use an atomic flag (e.g., AtomicBool with compare_and_swap) and only write to the pipe if the flag transitions from false to true, avoiding repeated lock acquisitions.

#### F118 · 🟡 medium · [performance] Stream buffer exhaustion under per-token backpressure · ⊘ WONT-FIX

- **Location:** `crates/infer-server/src/client/zmq_client.rs:23, 536`
- **Resolution:** ⊘ Buffer size is a tuning constant; the ZMQ client already re-arms the deadline on each chunk (zmq_client.rs:404), so slow-but-alive clients aren't falsely cancelled.

**Problem.** STREAM_CHUNK_BUFFER is set to 64, meaning the mpsc::Sender can queue at most 64 chunks before returning TrySendError::Full. At 50 tokens/second (typical inference speed), 64 chunks fills in ~1.3 seconds. If the HTTP client consumes slower (e.g., 10 tokens/sec), the buffer overflows immediately, causing `CancelReason::StreamTimeout` to be sent even though the request hasn't timed out. This creates false cancellations under normal backpressure.

```rust
const STREAM_CHUNK_BUFFER: usize = 64;
```

**Fix.** Increase the buffer size to a larger value (e.g., 256+) or make it configurable. Alternatively, use a dynamic backpressure mechanism that doesn't cancel the request prematurely.

#### F120 · 🟡 medium · [performance] Double deserialization per stream chunk in ZMQ handler · ⊘ WONT-FIX

- **Location:** `crates/infer-server/src/client/zmq_client.rs:349-436`
- **Resolution:** ⊘ Reordering msgpack deserialization by tag is risky (struct disjointness isn't guaranteed) and the cost (~µs) is dwarfed by the ZMQ recv syscall. Not worth the correctness risk.

**Problem.** handle_response() first tries to deserialize as InferenceResponse (line 349), and only if that fails does it try StreamChunk (line 398). This is two msgpack deserializations per chunk on the per-token hot path. Only one will ever succeed, making the first deserialize an unconditional waste of CPU on every streaming token.

```rust
if let Ok(response) = rmp_serde::from_slice::<InferenceResponse>(data) { ... }
        if let Ok(chunk) = rmp_serde::from_slice::<StreamChunk>(data) { ... }
```

**Fix.** Use a tagged enum or discriminator byte to determine the message type before deserializing, avoiding the second parse attempt.

#### F121 · 🟡 medium · [concurrency] Non-atomic deadline check before sending timeout chunk · ⊘ WONT-FIX

- **Location:** `crates/infer-server/src/client/zmq_client.rs:450-459`
- **Resolution:** ⊘ Single-threaded ZMQ client thread owns `pending`; the 'TOCTOU' is within one thread's sequential code, so no actual race.

**Problem.** cancel_timed_out_requests() reads the deadline from pending_req, compares it to now, and then removes from the map. Between the check and removal, another thread (or the ZMQ thread after receiving a Done chunk) could modify or remove the entry, causing a double-send of timeout or a lost timeout signal. This is a TOCTOU race on the pending map.

```rust
.filter_map(|(request_id, pending_req)| match pending_req {
                PendingRequest::Stream { deadline, .. } if *deadline <= now => {
                    Some((request_id.clone(), true))
                }
```

**Fix.** Collect the timed-out request IDs first, then in a second pass remove them and send timeouts, or use a lock-free collection.

#### F115 · ⚪ low · [performance] Blocking tokenizer encode/decode in async context

- **Location:** `crates/infer-server/src/api/openai/chat.rs:51-54`
- **Verified:** partial (severity high→low)

**Problem.** Synchronous tokenizer.encode() is called in an async handler without `.spawn_blocking()`. This is a blocking I/O call that will stall the async executor and delay all other requests on the worker pool. For a production inference server where tokenization can take milliseconds per request, this directly reduces throughput.

```rust
let encoding = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| AppError::internal(anyhow::anyhow!("Tokenize error: {}", e)))?;
```

**Fix.** Wrap the encode call in `tokio::task::spawn_blocking()` to move it off the async executor thread, or use an async-aware tokenizer implementation if available.

> **Verifier note:** Synchronous tokenizer.encode() is called in an async handler without spawn_blocking(). While technically this is a blocking operation, the practical performance impact is negligible for inference servers where tokenization (typically 1-10ms) is dwarfed by model inference time (100ms-seconds). The consistent pattern across all tokenizer calls (including decode operations in streaming paths) and the presence of tokenization monitoring suggests this is an intentional design choice to avoid spawn_blocking overhead for fast operations. Only becomes relevant if tokenization becomes a measured bottleneck.

#### F123 · ⚪ low · [design] Potential unwrap on user-provided token IDs

- **Location:** `crates/infer-server/src/api/openai/completion.rs:47-50`

**Problem.** When a user provides `CompletionPrompt::Tokens`, the code directly clones the Vec<i32> without validation. While not a direct unwrap, if the token IDs are invalid (e.g., out of vocab range), the subsequent inference will fail silently or produce garbled output. A validation step would catch user errors early.

```rust
CompletionPrompt::Tokens(ids) => {
            let len = ids.len() as u32;
            (ids.clone(), len)
        }
```

**Fix.** Add basic validation (e.g., check that token IDs are within vocab range) when accepting token arrays.


### infer-protocol · Wire structs + config resolution

#### F128 · 🟡 medium · [design] Potential dispatch inconsistency: resolve_model_type silent fallback to 'llama3' · ☑ FIXED

- **Location:** `crates/infer-protocol/src/config.rs:279-312`
- **Resolution:** ☑ `resolve_model_type` now warns on empty/unrecognized architecture hints before defaulting to llama3.

**Problem.** resolve_model_type returns 'llama3' as the default when no model_type or architectures field is found. This silent fallback could mask misconfigurations or unsupported model formats. If a config.json is corrupted, missing these fields, or uses an unexpected model architecture hint (e.g., 'mistral', 'bloom'), the function will still return 'llama3' instead of surfacing an error. This can lead to the wrong chat template or attention mask logic being applied.

```rust
let hint = cfg
    .get("model_type")
    .and_then(|v| v.as_str())
    .map(str::to_string)
    .or_else(|| {
        cfg.get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .map(str::to_string)
    })
    .unwrap_or_default()
    .to_lowercase();

let resolved = if hint.contains("qwen") {
    "qwen3"
} else {
    "llama3"
};
```

**Fix.** Either: (1) return an error if neither model_type nor architectures is found; (2) log a warning when falling back to 'llama3'; (3) add an explicit list of supported architectures and error on unknown types. This prevents silent misconfigurations that could produce incorrect inference results.

#### F129 · 🟡 medium · [performance] No validation that token_ids in AssignedIndices serialized size stays within bounds · ⊘ WONT-FIX

- **Location:** `crates/infer-protocol/src/worker_to_scheduler_data.rs:20-28`
- **Resolution:** ⊘ token_ids are only populated when prefix caching is enabled (opt-in); when on, they're the intended RadixTree payload. `is_consistent()` (F127) now guards malformed runs.

**Problem.** StepOutput.assigned_indices can contain Vec<i32> for each sequence, and during serialization these vectors grow the wire payload significantly. With 64 sequences in a batch and each having assigned_indices with all token_ids populated, the StepOutput serialized size could reach tens of KB per step. This is sent every decode step on the data plane and could impact throughput. The protocol places no size limit.

```rust
pub struct StepOutput {
    pub prefill_done: Vec<u64>,
    pub tokens: Vec<GeneratedToken>,
    /// 本步给每个 seq 新分配的全局 KV 索引段。`vec![]` 表示本步未分配
    /// 任何新槽位（例如 prefill 已结束、所有 seq 都在解码且无新增 KV）。
    #[serde(default)]
    pub assigned_indices: Vec<AssignedIndices>,
}
```

**Fix.** Consider conditionally populating token_ids in assigned_indices only when prefix-caching is enabled and RadixTree needs it. For non-prefix-caching mode, keep token_ids empty to reduce serialization overhead. Alternatively, add a validation in the scheduler that rejects StepOutputs exceeding a configurable size limit.

#### F127 · 🟡 medium · [correctness] Lack of overflow validation for token_ids.len() vs assigned_indices.len field · ☑ FIXED

- **Location:** `crates/infer-protocol/src/worker_to_scheduler_data.rs:39-41`
- **Verified:** partial (severity high→medium)
- **Resolution:** ☑ Added `AssignedIndices::is_consistent()` protocol helper; the scheduler consumer already warns+skips on mismatch (output_fns.rs:211).

**Problem.** The AssignedIndices struct documents that when token_ids is present, token_ids.len() must equal len. However, there is no runtime validation enforcing this invariant. A malicious or buggy worker could send token_ids with mismatched length. While len is u16 (max 65535), token_ids is Vec<i32> with unbounded length. This could lead to incorrect token-to-index binding in RadixTree or off-by-one errors in KV assignment.

```rust
pub struct AssignedIndices {
    pub sequence_id: u64,
    pub base: u32,
    pub len: u16,
    /// Token ids written into the KV slots represented by this run.
    ///
    /// Empty when prefix caching is disabled and the scheduler only needs slot
    /// accounting. When present, `token_ids.len()` must equal `len`.
    #[serde(default)]
    pub token_ids: Vec<i32>,
}
```

**Fix.** Add a validate() method on StepOutput or AssignedIndices that checks: (1) for each AssignedIndices where token_ids is non-empty, token_ids.len() == len as usize; (2) base + len does not overflow; (3) no two AssignedIndices for the same sequence_id overlap. Call this during deserialization or at protocol ingress.

#### F124 · 🟡 medium · [correctness] Integer overflow in AssignedIndices.end() with unconstrained indices · ☑ FIXED

- **Location:** `crates/infer-protocol/src/worker_to_scheduler_data.rs:45-47`
- **Verified:** partial (severity high→medium)
- **Resolution:** ☑ `AssignedIndices::end()` uses `saturating_add`; added `is_consistent()` to check overflow + token_ids/len match.

**Problem.** The end() method computes base + len without checking for overflow. While the worker should respect global KV pool limits, the protocol layer performs no validation on deserialized AssignedIndices. If a malicious or buggy worker sends base near u32::MAX with len > 0, the addition wraps silently, leading to incorrect block-table indexing downstream and potential memory corruption or slot allocation errors.

```rust
impl AssignedIndices {
    pub fn end(&self) -> u32 {
        self.base + self.len as u32
    }
}
```

**Fix.** Use saturating_add() or checked_add() to detect overflow: `self.base.checked_add(self.len as u32).ok_or_else(|| ...). Alternatively, add explicit validation in StepOutput processing to reject any AssignedIndices where base + len > max_kv_indices.

> **Verifier note:** Integer overflow CAN occur in AssignedIndices.end() when base + len wraps u32::MAX, but only in extreme scenarios: KV pool configured with ~4B blocks AND indices allocated at the high end with maximum-length runs (65535). In such a configuration, a long run at the end would overflow. The consequence is a budget leak (stale KV indices not freed to worker), not memory corruption. The code lacks defensive validation but the worker is a trusted component that cannot realistically produce this state in normal operation. The affected code path is the stale-sequence sanitization slow path, not the hot path.

#### F125 · 🟡 medium · [performance] Unnecessary String clones on hot path (per-token streaming) · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/application/output_fns.rs:265-266`
- **Resolution:** ⊘ The per-token String is required by the `StreamChunk.request_id: String` wire type; an Arc<str> would still materialize a String at the wire boundary. Cost is dwarfed by per-token msgpack+ZMQ send.

**Problem.** For every generated token in streaming mode, the code constructs a StreamChunk with request_id cloned from delivery.external_id. This happens on the decode hot path (per generated token) across all active streams. The external_id is a String that may be 36+ bytes (UUIDs). Accumulating these clones per token for a batch of 64 sequences means significant heap allocation overhead on a per-token basis.

```rust
token_chunks.push((
    delivery.client_id,
    StreamChunk {
        request_id: delivery.external_id,
        chunk_type: ChunkType::Token,
        token_id: Some(outcome.token_id),
        finish_reason: None,
        metrics: None,
    },
));
```

**Fix.** Use Arc<String> or Cow<str> for request_id to avoid per-token cloning. Alternatively, pass request_ids by reference through the protocol layer, or batch multiple tokens under one chunk. This is a hot-path per-token allocation affecting throughput under high-batch scenarios.

#### F126 · 🟡 medium · [performance] Redundant clones of external_id and image data in diffusion output path · ⊘ WONT-FIX

- **Location:** `crates/infer-scheduler/src/application/output_fns.rs:338, 351-352, 357`
- **Resolution:** ⊘ Diffusion output path — deferred per current priorities (diffusion out of scope).

**Problem.** In process_diffusion_step_decoded, the code clones external_id (line 338), then clones it again when building InferenceResponse (line 361). Image data is also cloned twice: once from item.image to ImageOutput (lines 351-352), then implicitly when the response is sent. For large image payloads (potentially megabytes), these clones represent significant overhead. This is not a per-token path but still wasteful for every completed image request.

```rust
let external_id = seq.meta.external_id.clone(); // line 338
...
let images = item.image.iter().map(|image| ImageOutput {
    ...
    format: image.format.clone(),
    data: image.data.clone(),  // clones raw image bytes
})
...
let response = InferenceResponse {
    request_id: external_id,  // uses the cloned external_id
```

**Fix.** Move external_id into the response instead of cloning (use take() or move semantics). For image data, use Arc<Vec<u8>> or transfer ownership to avoid redundant copies. This is lower priority than token streaming but still relevant for throughput.


### Worker · Diffusion core (DEFERRED)

#### F023 · 🔴 critical · [correctness] Incorrect tensor stride calculation on reshape — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/pipeline.rs:293-297`
- **Verified:** confirmed

**Problem.** The view_raw call on line 293-297 attempts to reshape [1, c, h, w] to [c, 1, h, w] but supplies strides [h*w, h*w, w, 1]. This is mathematically incorrect. For [c, 1, h, w] strides should be [1*h*w, h*w, w, 1] (step through c dimension by h*w elements), not [h*w, h*w, ...]. The duplicate h*w in positions 0 and 1 means both c and batch dimensions point to the same stride, causing aliasing and incorrect memory access.

Lines 293-298:
```
let latent_5d = sample.view_raw(
    crate::domain::types::Shape::from_slice(&[c, 1, h, w]),
    crate::domain::types::Shape::from_slice(&[h * w, h * w, w, 1]).contiguous_strides(),
    sample.offset_elems(),
    true,
);```

**Fix.** For [c, 1, h, w] strides from row-major [1, c, h, w], calculate as [c*h*w, h*w, w, 1] (or verify the reshape intent—is this a true view or a relayout that needs an actual copy/permute operation?).

> **Verifier note:** The view_raw calls on lines 293-297 and 308-314 of pipeline.rs attempt to reshape tensors using computed strides from intermediate shapes. The intermediate shapes ([h*w, h*w, w, 1] and [c*h*w, h*w, w, 1]) do not represent the target shapes correctly. When contiguous_strides() is called on these shapes, it produces strides that do not correctly map the source tensor elements to the target tensor indices. This causes data corruption when downstream operations like patchify_into call to_host_vec() on the mis-strided tensors. The tensors are marked as contiguous (is_contiguous=true) even though they have strided layouts, causing linear memcpy downloads to read data in the wrong order.

#### F024 · 🔴 critical · [correctness] Identical incorrect stride calculation on model output reshape — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/pipeline.rs:308-313`
- **Verified:** confirmed

**Problem.** Same stride calculation bug as line 293. The reshape from [c, 1, h, w] back to [1, c, h, w] uses strides [c*h*w, h*w, w, 1].contiguous_strides(), which is still incorrect. The contiguous_strides() call suggests a misunderstanding—passing pre-computed non-contiguous strides then calling contiguous_strides() on them may produce unintended results.

Lines 308-313:
```
let mo_4d = model_out.view_raw(
    crate::domain::types::Shape::from_slice(&[1, c, h, w]),
    crate::domain::types::Shape::from_slice(&[c * h * w, h * w, w, 1])
        .contiguous_strides(),
    model_out.offset_elems(),
    true,
);```

**Fix.** Verify the reshape intent. If converting [c, 1, h, w] to [1, c, h, w], use a real permute/transpose operation or ensure strides match row-major order. If this is a view-only operation, document why and verify the stride arithmetic.

> **Verifier note:** Incorrect stride calculation on model output reshape (line 310-311) and sample reshape (line 295). Both calls misuse contiguous_strides() by passing manually-crafted stride values as if they were shape dimensions, causing the strides to be recalculated incorrectly. The views are marked contiguous despite having wrong strides, leading to memory corruption when copy_from reads using those strides. This occurs in the hot denoise loop.

#### F022 · 🟠 high · [performance] Per-step heap allocations in denoise hot loop — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/pipeline.rs:316-324`
- **Verified:** confirmed

**Problem.** Three heap allocations (neg_mo, next_sample) occur inside the denoise timestep loop (lines 316, 321), which executes once per denoising step. For a 9-step generation, this is acceptable, but these tensors should have been pre-allocated in pipeline_state and reused to maintain the zero-alloc hot path invariant documented in state.rs line 8. This pattern defeats the allocate-once design.

```rust
Line 316: `let mut neg_mo: Tensor<T, Cuda> = Tensor::zeros([1, c, h, w], device)?;`
Line 321: `let mut next_sample: Tensor<T, Cuda> = Tensor::zeros([1, c, h, w], device)?;`
```

**Fix.** Pre-allocate neg_mo and next_sample in PipelineState (sized to max capacity), and reuse them across denoise loop iterations.

> **Verifier note:** Per-step CUDA device heap allocations occur inside the denoise timestep loop (lines 316, 321), which runs once per denoising step. Two tensors (neg_mo and next_sample) are allocated and freed each iteration. For a 9-step generation (the default), this results in 18 device allocations where 2 pre-allocated tensors exist in PipelineState (noise_pred, latents_tmp) specifically sized for reuse. This violates the documented allocate-once-and-reuse design (state.rs line 8) and introduces device memory allocation overhead into the hot path. The tensors are currently unused (line 265), indicating the reuse was planned but not implemented.

#### F027 · 🟡 medium · [design] Hardcoded ADALN buffer dimension instead of constant — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/dit_block.rs:376`

**Problem.** Line 376 allocates scratch.adaln_silu with hardcoded dimension [1, 256], but the codebase defines ADALN_EMBED_DIM = 256 as a constant. Using the constant would be more maintainable and catch dimension mismatches at compile time if the constant is ever changed.

```rust
Line 291: `pub adaln_silu: Tensor<T, D>, // [1, 256]` (comment correct, but allocated as [1, 256]).
Line 376: `let mut adaln_silu = vp2(&scratch.adaln_silu, 1, 256)?;` should use ADALN_EMBED_DIM.
```

**Fix.** Replace `256` with `ADALN_EMBED_DIM` on line 376.

#### F028 · 🟡 medium · [correctness] Incorrect integer division in n_patches_max calculation — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/state.rs:69`
- **Verified:** confirmed (severity high→medium)

**Problem.** Line 69 computes `f_t = 1 / self.f_patch_size.max(1)` using integer division. For f_patch_size >= 2, this yields 1 / f_patch_size = 0 (integer truncation), which is incorrect. The correct calculation should be `self.capacity.max_latent_h() / (self.f_patch_size * ...)` or similar to compute the number of patch tokens in the frequency (temporal) dimension.

```rust
Line 69: `let f_t = 1 / self.f_patch_size.max(1);`
For f_patch_size=1: f_t = 1/1 = 1 (correct by accident).
For f_patch_size=2: f_t = 1/2 = 0 (integer division, incorrect).
```

**Fix.** Verify the intended formula. Likely should be `self.capacity.max_latent_h() / (self.f_patch_size * self.patch_size)` or `self.capacity.max_latent_w() / (self.f_patch_size * self.patch_size)`. Compare with how h_t and w_t are calculated on lines 70-71.

> **Verifier note:** Line 69 contains incorrect integer division: `f_t = 1 / self.f_patch_size.max(1)`. The formula should be `f_t = max_latent_f() / self.f_patch_size` (similar to how h_t and w_t compute spatial tokens). For f_patch_size>=2, the current formula yields 0 via integer division; the code is saved from producing 0 tokens only by the `.max(1)` clamp on line 72. The bug is masked in practice because: (1) Z-Image hardcodes f_patch_size=1, (2) latent frequency F=1, and (3) the .max(1) clamp prevents catastrophic undercount. However, the code is semantically incorrect and would fail for models with F>1 or f_patch_size>1.

#### F025 · ⚪ low · [design] Duplicate prompt embedding dump code — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/pipeline.rs:238-243`

**Problem.** Lines 238-241 and 241-243 contain identical conditional dumps of prompt_embeds. This appears to be accidental duplication, likely from copy-paste during debug code addition.

Lines 238-241:
```
if std::env::var("RUSTINFER_DUMP_PROMPT").is_ok() {
    super::dit_block::dump_tensor("prompt_embeds_rust", &prompt_embeds);
}
if std::env::var("RUSTINFER_DUMP_PROMPT").is_ok() {```
Immediate repetition.

**Fix.** Remove one of the two identical blocks (lines 241-243).

#### F026 · ⚪ low · [design] Duplicate RoPE embedding dump code in transformer forward — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/transformer.rs:482-488`

**Problem.** Lines 475-481 and 482-488 contain identical dumps and debug prints for RoPE embeddings (x_cos, x_sin, img_pos_ids). This is accidental duplication.

```rust
Lines 475-481 dump x_cos, x_sin, and print img_pos_ids/s_cap/s_img/n. Lines 482-488 repeat identical dumps and prints.
```

**Fix.** Remove lines 482-488 (the duplicate block).


### Worker · Diffusion aux (DEFERRED)

> The diffusion auxiliary modules (patchify, 3D RoPE, text encoder, timestep embedder, VAE decoder) implement complex tensor transformations for the Z-Image diffusion pipeline. Overall architecture is sound, but three critical soundness/correctness issues were identified: (1) uninitialized memory creation in patchify and VAE permute functions using Vec::with_capacity + unsafe set_len without proper initialization, violating Rust's memory safety invariants; (2) missing bounds validation in RoPE 3D position indexing that can panic if position values exceed cached ranges; (3) host-side D2H/H2D roundtrips in VAE attention permutation that could be optimized with CUDA kernels. The text encoder's sin_host variable is dead code from refactoring.

#### F030 · 🟠 high · [soundness] Uninitialized memory in VAE decoder permute functions — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/vae_decoder.rs:217-219, 248-250`
- **Verified:** confirmed

**Problem.** Both permute_bchw_to_bnc and permute_bnc_to_bchw use the unsafe pattern: Vec::with_capacity then set_len without initialization. This violates Rust's memory safety invariants. If the nested loops do not execute (e.g., if b=0 or c=0), uninitialized memory is exposed.

```rust
let mut out: Vec<T> = Vec::with_capacity(b * n * c);
unsafe {
    out.set_len(b * n * c);
}
// Later indexed without prior writes if loops skip
```

**Fix.** Use vec![T::default(); size] instead, or add bounds assertions before set_len.

> **Verifier note:** Both permute_bchw_to_bnc (lines 217-220) and permute_bnc_to_bchw (lines 248-250) use unsafe Vec::with_capacity() followed by set_len() without initialization. When any dimension (b, c, h, or w) is 0, the nested loops don't execute, leaving the vector with uninitialized bytes. This violates Rust's memory safety invariants. While practical diffusion inference requires non-zero dimensions, there are no runtime guards preventing malformed inputs. The vectors are later passed to Tensor::from_host_slice which copies the potentially uninitialized data to GPU memory.

#### F031 · 🟡 medium · [correctness] Missing bounds check on RoPE position indices — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/rope_3d.rs:112-118`
- **Verified:** confirmed (severity high→medium)

**Problem.** The position value read from pos_ids[token * 3 + axis] is cast to usize and used to index cos_cache/sin_cache without verifying it is within axes_lens[axis]. If pos >= axes_lens[axis], the slice access cos_cache[cache_base..cache_base + half_d] will panic at runtime with an index out of bounds error.

```rust
let pos = pos_ids[token * 3 + axis] as usize;
let cache_base = pos * half_d;
cos_host[out_base..out_base + half_d]
    .copy_from_slice(&cos_cache[cache_base..cache_base + half_d]);
```

**Fix.** Add a bounds check: if pos >= self.axes_lens[axis] { return Err(OpError::...) } before indexing the cache.

> **Verifier note:** The position value read from pos_ids[token * 3 + axis] is cast to usize and used to compute cache_base = pos * half_d (lines 112-118, 173-179). This index is then used to slice cos_cache[cache_base..cache_base + half_d] and sin_cache[cache_base..cache_base + half_d] without runtime bounds verification that pos < self.axes_lens[axis]. If pos >= axes_lens[axis], the slice access will panic with an index out of bounds error. However, current callers use fill_cap_pos_ids and fill_image_pos_ids which produce valid positions, so the vulnerability is not presently triggered in practice. The actual impact is mitigated by caller discipline, but the functions lack defensive bounds checking despite having explicit validation elsewhere, making this a correctness gap rather than an exploitable runtime risk in the current codebase.

#### F033 · 🟡 medium · [correctness] Missing position value bounds validation in RoPE embed_into_cuda — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/rope_3d.rs:78-84`

**Problem.** The embed_into_cuda function checks pos_ids.len() == seq_len * 3 but does not validate that the position values themselves are within the cached ranges. Combined with the missing bounds check at line 112, this allows out-of-bounds cache accesses.

```rust
if pos_ids.len() != seq_len * 3 {
    return Err(...);
}
// But no check that each pos_ids[i] < axes_lens[i % 3]
```

**Fix.** Add validation loop: for i in 0..seq_len { for axis in 0..3 { if pos_ids[i*3+axis] as usize >= axes_lens[axis] { return Err(...) } } }

#### F034 · 🟡 medium · [performance] Inefficient host roundtrip in VAE attention permutation — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/vae_decoder.rs:207-235, 238-264`

**Problem.** The permute_bchw_to_bnc and permute_bnc_to_bchw functions do D2H transfers for every VAE attention block. In VAE decoder, the attention block operates on mid-block features (up to 512 channels × 16×16 spatial = ~131k elements per batch). For BF16, each permute is ~260KB of GPU-CPU roundtrip traffic. This runs on the hot path (per-image decode) and could be optimized with a CUDA kernel.

```rust
let host = x.to_host_vec()?;
let mut out: Vec<T> = Vec::with_capacity(...);
// permute on host
Tensor::from_host_slice(&out, ...)
```

**Fix.** Implement a fused CUDA permutation kernel to avoid host roundtrip, especially if batching or multi-image inference is used.

#### F029 · ⚪ low · [soundness] Uninitialized memory access in patchify vec_uninit — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/patchify.rs:165-170`
- **Verified:** partial (severity high→low)

**Problem.** Vec::with_capacity(n) does not initialize memory. Calling set_len(n) without initialization creates uninitialized memory that is then cast to valid elements. While the code claims it writes every element before reading, the pattern violates Rust's safety invariants. The comment incorrectly states 'zero-init by default' when with_capacity does no such thing.

```rust
let mut v: Vec<T> = Vec::with_capacity(n);
unsafe {
    v.set_len(n);
}
```

**Fix.** Use vec![T::default(); n] or properly document why MaybeUninit-based initialization is needed. The current comment is misleading.

> **Verifier note:** The code uses unsafe set_len after with_capacity, creating uninitialized memory, but the pattern is sound because: (1) every vector element is guaranteed to be written by the caller before reading, (2) T is constrained to Copy (no Drop impl), and (3) the loops exhaustively cover all indices. The real issue is the comment at line 161 is misleading (says "zero-init by default" when with_capacity doesn't do that), but the SAFETY justification (line 167-168) is correct.

#### F032 · ⚪ low · [design] Dead code: unused sin_host in text encoder — **DEFERRED (diffusion)**

- **Location:** `crates/infer-worker/src/models/diffusion/text_encoder.rs:191, 230`

**Problem.** Line 191 creates sin_host as vec![T::DATA_TYPE; 0], which creates an empty vector with a suspicious type annotation. Line 230 then marks it unused with 'let _ = sin_host'. This appears to be leftover from a refactor and serves no purpose.

```rust
let sin_host = vec![T::DATA_TYPE; 0];
...
let _ = sin_host; // unused
```

**Fix.** Remove the sin_host variable entirely (lines 191 and 230).

---

## Appendix — Refuted findings (checked, NOT real)

Flagged by a reader but refuted by an independent verifier that read the actual code. Recorded so they are not re-investigated.

| Original sev | File:Lines | Claim | Why refuted (short) |
|---|---|---|---|
| high | `crates/infer-core/src/ports/fused_ops.rs:557-561` | rmsnorm_heads may compute out-of-bounds weight indexing | The claim alleges an out-of-bounds indexing vulnerability in rmsnorm_heads (lines 557-561). However, the actual code structure makes this impossible:  (1) Line 532-540 validates weight_len with an explicit check: `if wei… |
| high | `crates/infer-core/src/ports/math_ops.rs:161-165` | bitcast may accept misaligned offset without validation | The bitcast alignment check is sufficient for current supported dtypes and device allocators. However, the suggested documentation would be valuable for maintainability: if future dtypes with larger SIZE_BYTES (e.g., 16+… |
| high | `crates/infer-core/src/tensor.rs:45, 184` | Integer overflow in shape.numel() is not protected in view operations | The claim alleges that view_raw (line 184 in tensor.rs) lacks overflow protection that from_raw_parts (line 45 in tensor.rs) has. However, reading the actual code reveals: (1) from_raw_parts calls shape.numel() at tensor… |
| high | `crates/infer-backend-cuda/src/kernels/matmul/sdpa.rs:222-256` | Unbounded pointer arithmetic in GQA KV replication | The claim alleges unbounded pointer arithmetic with no validation in the GQA KV replication loop. However, examining the actual code reveals:  (a) The cited code does perform pointer arithmetic via `.add(src_off)` and `.… |
| high | `crates/infer-backend-cuda/src/kernels/sampler/mod.rs:25-38` | FFI argument binding mismatch: F16/F32 argmax signatures incomplete | F16 and F32 argmax bindings correctly have 5 parameters while BF16 has 7, reflecting the intentional design decision that F16/F32 do not yet support selective row argmax. This is documented in code comments and enforced … |
| high | `crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:117, 241` | Potential pointer overflow in pad.rs due to unchecked multiplication | The claim alleges unchecked multiplication overflow at lines 117 and 241 in pad.rs. Upon examination:  Line 117 (/data/home/vinciiliu/RustInfer/crates/infer-backend-cuda/src/kernels/cast_fill/pad.rs:117): ```rust let dst… |
| high | `crates/infer-backend-cuda/src/kernels/qkv_norm_rope_scatter/mod.rs:93-98` | Incomplete fix for positions array overflow in qkv_norm_rope_scatter | The claim is based on a misunderstanding of the code's structure and the nature of the fix. The code is actually correct:  1. **Dimension consistency is guaranteed by design**: In lib.rs:563-565, q, k, and v are all crea… |
| high | `crates/infer-backend-cuda/src/kernels/softmax/mod.rs:41` | Unchecked division for row count calculation in softmax and fused kernels | The claim alleges that unchecked division in softmax (line 41 of softmax/mod.rs) and fused_add_rmsnorm (line 52-53 of fused_add_rmsnorm/mod.rs) could produce incorrect row counts. However, this analysis is flawed:  MATHE… |
| high | `crates/infer-scheduler/src/application/batch_builder.rs:172-175` | O(n*m) prefix hint lookup in hot path | The claim of "O(n*m) prefix hint lookup in hot path" misrepresents the actual complexity bounds. Code analysis shows:  1. **Bounded sizes (not arbitrary n, m)**: Lines 169-171 of batch_builder.rs explicitly document that… |
| high | `crates/infer-scheduler/src/application/workflow/llm.rs:240-270` | KV slot count race between calculation and removal in non-prefix mode | The claim alleges a race condition where `finished_kv_slots` calculated at llm.rs:246-254 becomes stale because state modifications in `process_llm_step_decoded()` (specifically `append_generated_token()` or `finish_deco… |
| critical | `crates/infer-scheduler/src/domain/policy/continuous_batching.rs:120-121, 165-167` | Off-by-one error in sequence admission budget check | The claim is mathematically incorrect. Reading the code at /data/home/vinciiliu/RustInfer/crates/infer-scheduler/src/domain/policy/continuous_batching.rs:  For the SJF path (lines 120-161): The loop uses `enumerate()` wh… |
| high | `crates/infer-scheduler/src/infrastructure/transport/control_plane/pending_calls.rs:117-142, 179-181` | Race condition in PendingCalls AllRx completion with concurrent sweep_expired | No race condition exists. The router thread runs a strictly sequential state machine: first inbound frame processing (calling complete()), then command processing, then deadline sweep (calling sweep_expired()). The std::… |
| high | `crates/infer-scheduler/src/infrastructure/transport/control_plane/router_thread.rs:255-273` | Race between liveness eviction and late heartbeat resurrection | The code does NOT have the described race condition. The liveness watchdog removes evicted workers from the shared registry view under a write lock, and when the router thread's `handle_inbound()` later calls `intern()` … |
| high | `crates/infer-scheduler/src/infrastructure/transport/zmq_transport.rs:126-154` | ZMQ ROUTER frame handling lacks validation for DEALER/REQ asymmetry | The claim misunderstands ZMQ ROUTER socket semantics and the guarantees provided by zmq_poll().   Evidence from actual code: - /data/home/vinciiliu/RustInfer/crates/infer-scheduler/src/infrastructure/transport/zmq_transp… |
| high | `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:528-632` | Race condition in split_edge between owners computation and final state | The claim alleges that an owner with tip.pos == pos can appear in suffix_transitive_owners (via inherited children's owner sets) and thus be added to suffix_owners, yet never have its tip updated to point to the suffix (… |
| high | `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:437-438` | Insufficient condition for LRU eviction: children.is_empty() check can fail with lazy removal | No issue found. The code correctly implements LRU eviction with proper parent promotion. The check at line 438 is defensive programming that validates the LRU entry is still valid after potential stale entries were skipp… |
| high | `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs:247-254` | ChainTip.pos bounds check missing before array access in append_token | The claim alleges that tip.pos can reach edge_tokens.len() and then cause an out-of-bounds access via edge_tokens[tip.pos] in subsequent append_token calls. However, examination of the code reveals a correct bounds-check… |
| high | `crates/infer-server/src/api/openai/decoder.rs:56-60` | Blocking tokenizer decode in streaming hot loop | The current IncrementalDecoder implementation has already been optimized (commit 3f3381d) to limit decoding to a small bounded window (MAX_PENDING_TOKENS=16) and clears the buffer after each token. The decode operation i… |
| critical | `crates/infer-worker/src/infrastructure/io/safetensors.rs:34` | Unsound mmap lifetime cast in SafetensorsReader | The claim mischaracterizes the actual code. At /data/home/vinciiliu/RustInfer/crates/infer-worker/src/infrastructure/io/safetensors.rs line 156, the PUBLIC API `read_view()` returns `TensorView<'_>` with an elided lifeti… |
| high | `crates/infer-worker/src/infrastructure/io/safetensors.rs:144` | Panic on malformed safetensors index.json weight_map | The claim mischaracterizes the code flow. While line 144 does call `.unwrap()`, it is protected by validation that occurs in the preceding loop (lines 125-133). The first iteration validates every entry in weight_map by … |
| high | `crates/infer-worker/src/infrastructure/io/safetensors.rs:162` | No validation of shard index bounds when routing read_view calls | The claim alleges that line 162 in safetensors.rs (`self.shards[idx]`) will panic if `idx` is out of bounds. However, examining the code reveals a maintained invariant:  1. In `open_sharded()` (lines 112-152), the `name_… |
| high | `crates/infer-worker/src/application/runtime.rs:2437-2440` | D2H copy in issue_decode_abc over-reads beyond compacted active token range | The claim alleges a soundness bug where D2H copy at lines 2437-2440 over-reads beyond valid data. However, careful analysis of the code shows: (1) input_ids_buf is allocated with cap_batch elements (line 293), (2) batch … |
| high | `crates/infer-worker/src/application/decode_engine.rs:854-858` | Dangerous unwrap on decode order consistency after control drain | The claim assumes control messages drain between `prepare_step` and `build_decode_request` in `issue_new`, but the actual code structure prevents this. Key evidence:  1. FILE:/data/home/vinciiliu/RustInfer/crates/infer-w… |
| high | `crates/infer-worker/src/application/worker_scheduler.rs:206-213` | EOS/finished row index bounds not validated for fused mixed outputs | The claimed vulnerability assumes that row indices computed as `token_offset + local` in `append_prefill_abc_next_rows` (lines 206-214) could go out-of-bounds when indexing `out_finished`. Analysis of the actual code rev… |
