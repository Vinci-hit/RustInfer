# Modular-Backend Refactor — Implementation Status

Tracks execution of `docs/modular-backend-architecture-proposal.md` on branch
`feat/worker-batch-forward`. The proposal's prime directive is **always-green,
incremental**: the build is green at the end of every step and the
`default = ["cuda"]` flip happens last.

Verification gate used throughout: the **no-CUDA core** must stay green
(`cargo check -p infer-worker --no-default-features`, ~0.6 s) and the CUDA path
must stay green (`cargo check -p infer-worker`, kernels cached). The CUDA build
here relies on a conda CUDA 12.8 toolchain discovered by `build.rs`.

## Landed

### Stage 0 — reconcile the two `Device` traits ✅
The codebase had **two** distinct public traits named `Device`
(`domain/ports/device.rs` with `type ExecCtx`, and `domain/exec.rs` with
`type Scope: ExecScope`). Per the proposal's Stage 0, the exec-layer trait was
renamed to **`ExecDevice`** (and `exec::HostDevice` → `ExecHostDevice`), so the
eventual `infer-core` public surface carries a single `Device`. Bare importers
were kept working by import-aliasing (`use …::ExecDevice as Device;`), so no
call sites changed semantics.

- Files touched: `domain/exec.rs` (trait defs + bounds), and qualified refs
  across `domain/ports/*`, `domain/kv.rs`, `infrastructure/cpu/mod.rs`,
  `infrastructure/cuda/mod.rs`, `application/runtime.rs`.
- Verified: no-CUDA core green, CUDA check green, **all 55 worker tests pass**.

### Stage 1a — extract the `infer-core` foundation crate ✅
First physically-extracted slice of the foundation DAG (proposal §1). A new
`rustc`-only crate `infer-core` now owns the clean, backend-independent leaf:

- `types` — `Dims`/`Shape`/`Strides`, `DataType`, the storage-`Dtype` trait, `Float`.
- `dtype` — the extended `DTypeId` (+ runtime registry ≥1024), read/write `Dtype`,
  `Fp8E4m3`/`Fp8E5m2`, and `quant`.
- `error` — `OpError`/`OpResult`.
- `device` — `MemoryPort`, `Device` (the type-state port), `HostDevice`,
  `Allocator` — the already-half-built raw-bytes seam.
- `storage` — `Storage<D>`, the sole RAII owner of `MemoryPort` allocations.
  (Its `Cpu`-coupled tests were relocated into the worker's CPU test module.)
- `tensor` — `Tensor<T, D>` itself, the monomorphized type-state tensor (incl.
  the host-generic `as_slice`/`as_slice_mut`). Two enabling changes:
  (1) the Cpu constructors `zeros_cpu`/`from_slice` became an extension trait
  `CpuTensorExt` in the worker (a crate can't add inherent methods to a foreign
  type); (2) its ~260 lines of `Cpu`/`Cuda` tests moved to
  `domain/tensor_tests.rs` in the worker; (3) the one struct-literal build site
  (`MathOps::bitcast`) now uses the new public `Tensor::from_raw_parts`.

`infer-worker` depends on `infer-core` and re-exports each module
(`pub use infer_core::{dtype, types};`, `pub use infer_core::error;`,
`pub use infer_core::device;`) from `domain/`, so every existing
`crate::domain::{types,dtype,device,…}` / `…::ports::error` path resolves
unchanged. Verified: both gates green, **all 55 worker tests pass**.

- `exec` — the full exec vocabulary: `ExecScope`, `ExecDevice`, `Stream`,
  `Workspace<D>`, `ActiveGuard<D>`, `StepCtx<D>`, `Rank`/`TopologyShape`,
  `DeviceId`, `MaskHandle`, `QuantTier`, and `CommAxis` (moved here from the
  `CollectiveOps` trait, its only structural user, and re-exported by the
  worker's `collective`). `infer-core` gained an `infer-protocol` dep for
  `TopologyShape::from_load_model`.
- `plan` — the `BatchPlan` cluster (`BatchPlan`/`BatchKind`/`MaskMode`/
  `RAGGED_Q_TILE`) that `StepCtx` borrows. The runtime plan/request/response
  types (`SeqStep`/`StepRequest`/`StepOutput`/…) stay in the worker's
  `domain::plan`, which re-exports the cluster.

**The recursive-middle collapse the proposal called for is done** — the entire
foundation (value types, dtype, error, the device/memory seam, `Storage`,
`Tensor`, and the exec vocabulary) now lives in `infer-core` and builds with
`rustc` alone.

### Stage 2 — op-port traits → `infer-backend-abi` ✅
The backend **interface** crate now exists (rustc-only, depends on `infer-core`):
`ports` (the op-port traits `CoreOps`/`MathOps`/`FusedOps`/`DiffusionOps`/
`CollectiveOps`/`OpBackend`/`Backend`/`Sampler` + the `impl_math_ops_via_core_ops!`
macro), `kv` (the paged KV pool + `KvView`/`LayerKv` view types that appear in
`FusedOps` signatures), and `component` (`Component`/`Hidden`/`LayerRange`/
`StageKind`). `SampledToken` moved here too (next to the `Sampler` interface).

This was **one atomic move** — `component` ↔ `kv` ↔ `ports` are mutually
recursive (`FusedOps` takes `KvView`; `KvView`'s `LayerKv` holds `&mut PagedKvPool`;
`PagedKvPool` needs `component::LayerRange`; `component` needs `kv::KvView` +
`ports::backend::LlmBackend`). Mechanics: ~10 files relocated; `crate::domain::*`
paths rewritten (`dtype`/`types`/`tensor`/`exec`/`storage`/`plan` → `infer_core::*`,
`ports`/`kv`/`component` → `crate::*`); the macro's 145 `$crate::domain::*` paths
rewritten; the `kv` `Cpu`-coupled tests relocated to the worker's
`domain/kv_tests.rs`; the worker re-exports `infer_backend_abi::{ports,kv,component}`
so every `crate::domain::…` path resolves unchanged. Verified: both gates green,
**55 worker tests pass**, `infer-backend-abi` builds warning-free.

### Stage 3 — backends → their own static crates ✅
Both adapters are now independent, statically-linked crates:

- **`infer-backend-cpu`** — `infrastructure/cpu` → its own crate (deps:
  `infer-core` + `infer-backend-abi`, `rustc`-only). The always-linked baseline.
  Its ~16 CPU op/storage tests run as `cargo test -p infer-backend-cpu`.
- **`infer-backend-cuda`** — the entire `infrastructure/cuda` tree (5 `.rs` +
  28 kernel wrappers + 29 `.cu` + CUTLASS + `wrapper.h`) **and the 471-line
  `build.rs`** → its own crate. The CUDA toolchain (nvcc, bindgen/libclang,
  cuDNN, cc) is now confined entirely here; `build.rs` paths rewired
  (`src/infrastructure/cuda/*` → `src/*`) and its vestigial `cfg(feature="cuda")`
  gates removed (the crate *is* the CUDA backend). Builds standalone in ~1m24s
  (kernels + bindgen + Rust).
- One worker tendril resolved: `env_flags` (a std-only leaf) moved to
  `infer-core` so `infer-backend-cuda` can read `force_gemm()` without depending
  on the worker.

**The worker no longer has a `build.rs` and no longer carries `cc`/`bindgen`/
`walkdir`.** Its `cuda` feature went from `["dep:cc","dep:bindgen","dep:walkdir"]`
(compile kernels in-tree) to `["dep:infer-backend-cuda"]` (link the static
backend crate). `infrastructure/mod.rs` re-exports `infer_backend_cpu as cpu`
(always) and `infer_backend_cuda as cuda` (under the `cuda` feature), so all
`crate::infrastructure::{cpu,cuda}::…` paths resolve unchanged.

Verified: `cargo check -p infer-worker --no-default-features` builds the core +
CPU backend **with `rustc` alone — no nvcc, no build script** (the original §1
pain point, eliminated); the CUDA path (`cargo check -p infer-worker`) links the
isolated `infer-backend-cuda` and is green; 39 worker + 16 CPU-crate tests pass.

**The static multi-backend goal is met.** Crate DAG:
`infer-core` ← `infer-backend-abi` ← {`infer-backend-cpu`, `infer-backend-cuda`}
and `infer-worker` selects backends by Cargo feature, fully monomorphized, with
zero runtime dispatch.

### Stage 4 — runtime/models peel: assessed, deferred (Cuda-entangled)
The remaining `application/*` + `models/*` peel is **not a clean code-org move**.
Two real couplings make it a refactor rather than a relocation:

- **`serve_loop` is hardcoded to `Cuda`** (`M: DecoderModel<bf16, Cuda>`, `&Cuda`,
  `CudaScope`). Peeling it into a backend-agnostic `infer-runtime` requires making
  it generic over the backend `D` and selecting the concrete backend in the
  worker bin — a genuine refactor, not a `git mv`.
- **The diffusion models are Cuda-specific by design** (the project's "diffusion
  runs on CUDA only" decision): ~295 non-test lines in `models/diffusion/*` take
  `&Tensor<T, Cuda>` directly. So `infer-models` cannot be backend-agnostic
  without either feature-gating diffusion behind `infer-backend-cuda` or
  rewriting it generic.

Because the **architecture objectives are already met** (backends isolated as
static crates, GPU-free core builds with `rustc` alone, full monomorphization,
zero runtime dispatch), this peel is **optional organizational polish** best done
as a focused follow-up. The LLM model path (`decoder`/`layers`/`llama3`/`qwen3`)
is generic and could extract cleanly; the diffusion + serve-loop path needs the
generic-over-backend refactor first.

Likewise, flipping `default` from `["cuda"]` to `[]` is a one-liner, but the
worker **bin** is still `required-features = ["cuda"]` (no CPU serve loop exists
yet), so a GPU-free `default` only changes which features a bare `cargo build`
enables for the lib — the lib is already provably GPU-free via
`--no-default-features`. Deferred until a CPU serve path exists.

### The stable C-ABI seam — `infer-abi` (proposal §2, §4) ✅
A new, `rustc`-only workspace crate implementing the loadable-backend boundary —
the C-ABI half of the proposal's `infer-backend-abi`. No GPU toolchain.

- `DeviceTensor` — `#[repr(C)]`, borrowed, DLPack-shaped: `u16` dtype tag,
  `byte_offset`, `MAX_RANK = 8` inline shape/strides (element strides),
  `owner_backend_id`. Never owns; host keep-alive contract documented.
- `DTypeId` (`u16`, `>= 1024` registry range), `StreamHandle`, `Capabilities`
  (feature-mask split from hardware compute-cap gate), `AbiStatus` (`i32`) +
  `ErrorBuf` transport.
- `BackendVTable` — `#[repr(C)]` struct-of-`extern "C"`-fn-pointers with
  `struct_size`/`abi_version` first and **append-only** fields; grouped
  identity → lifecycle → memory → compute → error → extension-chain.
- `rustinfer_get_backend_api(abi_version) -> *const BackendVTable` entry
  contract; `ffi_guard` panic boundary; `check_compatible` version gate.
- `loader::Registry` — owned/droppable, `dlopen` via `libloading`, priority
  ordering, `RUSTINFER_BACKENDS=name:path;…` discovery, `best_for(op,dtype,cap)`
  selection, `Library`-outlives-handles.
- Tests: 8 POD/version-gate unit tests + 4 integration tests (in-process
  `register_static`, full alloc→upload→`op_add`/`op_matmul`→download round-trip,
  wrong-`owner_backend_id` rejection, major-version refusal, **real `dlopen` of
  the cdylib**). All green.

### Reference backend — `infer-backend-cpu-ref` ✅
A complete, GPU-free `cdylib + rlib` plugin implementing the full vtable
(system-allocator memory, scalar reference kernels that re-monomorphize on the
`dtype_tag`, every entry wrapped in `ffi_guard`). Proves the ABI works
end-to-end and is the `dlopen` target for `infer-abi`'s integration test.
Artifact: `target/debug/libinfer_backend_cpu_ref.so`.

## Build / test

```bash
cargo build -p infer-core                            # GPU-free foundation crate
cargo check -p infer-worker --no-default-features   # no-CUDA core gate (~0.6s)
cargo check -p infer-worker                         # CUDA path (kernels cached)
cargo test  -p infer-worker --no-default-features    # 55 CPU tests
cargo build -p infer-backend-cpu-ref                 # build the plugin cdylib
cargo test  -p infer-abi                             # 8 unit + 4 integration (incl. dlopen)
```

### Crate consolidation — 11 → 8 ✅
After the static decision, the crate set was trimmed to the minimal layout:

- **Deleted** `infer-abi` + `infer-backend-cpu-ref` — the shelved C-ABI vtable/
  loader + its reference plugin. They implemented the *dynamic* path, which is
  not part of the static design, and had zero dependents.
- **Merged** `infer-backend-abi` → `infer-core`. With no C-ABI boundary, the
  op-port traits (`ports`/`kv`/`component` + the `impl_math_ops_via_core_ops!`
  macro) have no reason to live in a separate crate, so they fold into the
  foundation. `infer-core` uses `extern crate self as infer_core;` so the moved
  modules and the macro keep their `infer_core::…` / `$crate::…` paths; backends
  now depend on **one** crate (`infer-core`) instead of two.

Final workspace (8 crates):
`infer-protocol · infer-core (foundation + op-port traits) · infer-backend-cpu ·
infer-backend-cuda · infer-worker · infer-scheduler · infer-server ·
infer-frontend`. Verified: both gates green, 39 worker + 16 CPU-crate tests pass.

## Not yet landed (remaining proposal stages)

These are the larger, higher-risk stages. They were deliberately **not** done
blind because the only fast green gate available here is the no-CUDA core; a full
worker crate-split without a tight CUDA build loop would risk leaving the working
branch broken — violating the always-green rule.

- **Stage 1 (rest)** — `infer-core` now exists (leaf + device ports, above);
  remaining is to migrate the recursive middle (`exec`/`Storage`/`Tensor` + the
  op-port traits, relocating their `Cpu`/`Cuda` tests) into `infer-core` /
  `infer-backend-abi`, peel `application/`+`models/` into `infer-runtime` /
  `infer-models`, and move the 29 `.cu` into a static-linked `infer-backend-cuda`
  crate (CUDA still static, `default = ["cuda"]` unchanged).
- **Stage 2** — `BackendDyn` erased trait + per-dtype re-mono adapter +
  CPU-capable `serve_loop`; exit criterion `cargo test -p infer-runtime
  --no-default-features` runs a model on CPU.
- **Stage 3** — wire `infer-abi`'s `Registry` into the worker host; ship
  `infer-backend-cuda` as a cdylib via `cudarc` dynamic-loading; per-op arg
  structs for the fused/paged ops on the extension chain.
- **Stage 4** — flip `default` to `[]`, split distribution trains, GPU-probing
  install CLI.
