# RustInfer — Crate Architecture

This document describes the workspace after the static modular-backend refactor:
what each crate is, what it owns, how they depend on one another, and how a
backend is selected. It is the reference companion to
`modular-backend-architecture-proposal.md` (the design rationale) and
`modular-backend-implementation-status.md` (the staged execution log).

The guiding principle: **multiple backends, selected at compile time, fully
monomorphized.** `Tensor<T, D>` carries the device type `D` from the runtime
down into each backend's kernels; there is no runtime dispatch, no vtable, and
no C ABI on the inference path. A backend is an ordinary Rust crate that
implements a set of traits and is linked via a Cargo feature. The core builds
with `rustc` alone when no GPU backend is selected.

## Workspace at a glance

Eight crates (all edition 2024 except the WASM UI). New or substantially
modified by the refactor are marked **★**.

| Crate | Role | Depends on (in-workspace) | Toolchain to build |
|---|---|---|---|
| `infer-protocol` | shared wire types (ZMQ messages, config) | — | rustc |
| **★ `infer-core`** | foundation **+** backend interface | `infer-protocol` | rustc |
| **★ `infer-backend-cpu`** | CPU backend (reference, always linked) | `infer-core` | rustc |
| **★ `infer-backend-cuda`** | CUDA backend (kernels + toolchain, isolated) | `infer-core` | CUDA (nvcc, bindgen/libclang, cuDNN) |
| **★ `infer-worker`** | inference runtime + process host | `infer-protocol`, `infer-core`, `infer-backend-cpu`, `infer-backend-cuda`* | rustc (CUDA only if `cuda` feature on) |
| `infer-scheduler` | request scheduler process | `infer-protocol` | rustc |
| `infer-server` | HTTP/SSE OpenAI-compatible API process | `infer-protocol` | rustc |
| `infer-frontend` | Dioxus WASM UI (edition 2021, standalone) | — | wasm target |

\* `infer-backend-cuda` is an **optional** dependency of `infer-worker`, pulled
in only by the `cuda` feature.

## Dependency DAG

```
                         infer-protocol
                       (dependency-free wire types)
                        ↑      ↑        ↑      ↑
        ┌───────────────┘      │        │      └───────────────┐
   infer-scheduler        infer-core    │                 infer-server
   (scheduler proc)    (foundation +    │                 (HTTP API proc)
                        op-port traits)  │
                            ↑    ↑       │
            ┌───────────────┘    └───────────────┐
   infer-backend-cpu                      infer-backend-cuda
   (CPU, always linked)                   (CUDA, feature-gated, toolchain here)
            ↑                                     ↑
            └──────────────┬──────────────────────┘
                     infer-worker
            (runtime + ZMQ host; selects backend by Cargo feature)

   infer-frontend  ── standalone WASM UI (no in-workspace deps)
```

There are no back-edges: nothing in `infer-core` or the backend crates depends
on `infer-worker`. The only crate that needs the CUDA toolchain to build is
`infer-backend-cuda`, and it is a leaf.

## The three runtime processes

Inference runs as three processes coordinated over ZMQ IPC, all driven by one
TOML config (`--config`):

- **`infer-scheduler`** — continuous-batching scheduler + KV-allocation
  orchestration. Talks to the worker (data/control) and the server.
- **`infer-worker`** — the GPU runtime: loads weights, owns the paged KV pool,
  runs the decode loop on the selected backend. Binary: `rustinfer-worker`.
- **`infer-server`** — axum HTTP server exposing the OpenAI-style
  `/v1/completions` + `/v1/models`. Binary: `rustinfer-server`.

Start order: scheduler → worker → server (see `scripts/start_*.sh`).

---

## ★ `infer-core` — foundation + backend interface

The bottom of the DAG and the single crate every backend implements against. It
builds with `rustc` alone (deps: `infer-protocol`, `half`, `thiserror`). It uses
`extern crate self as infer_core;` so its own modules — and the
`impl_math_ops_via_core_ops!` macro — can refer to the crate uniformly as
`infer_core::…` whether the code is compiled here or expanded inside a backend.

It has two halves.

**Foundation** (the monomorphized data layer):
- `types` — `Dims`/`Shape`/`Strides`, `DataType`, the storage-`Dtype` trait, `Float`.
- `dtype` — the extended `DTypeId` (runtime registry, ≥1024 range), the read/write
  `Dtype` trait, `Fp8E4m3`/`Fp8E5m2`, and `quant` (`QuantScheme`).
- `error` — `OpError` / `OpResult`.
- `device` — `MemoryPort` (raw-byte alloc/upload/download/copy), `Device` (the
  type-state device port), `HostDevice`, `Allocator`.
- `storage` — `Storage<D>`, the sole RAII owner of `MemoryPort` allocations.
- `tensor` — `Tensor<T, D>`, the monomorphized type-state tensor (plus the
  host-generic `as_slice`/`as_slice_mut` for `HostDevice` backends, and the
  `Tensor::from_raw_parts` public constructor).
- `exec` — `ExecDevice` (the exec-layer device carrying `type Scope`), `ExecScope`,
  `Stream`, `Workspace<D>`, `ActiveGuard<D>`, `StepCtx<D>`, `Rank`/`TopologyShape`,
  `DeviceId`, `MaskHandle`, `QuantTier`, `CommAxis`.
- `plan` — the `BatchPlan` cluster (`BatchPlan`/`BatchKind`/`MaskMode`/`RAGGED_Q_TILE`)
  that `StepCtx` borrows.
- `env_flags` — process-wide flags read once from the environment (e.g.
  `force_gemm`, `disable_graph`).

**Backend interface** (the compile-time seam — Rust traits, no C ABI):
- `ports` — the op-port traits a backend implements:
  - `CoreOps`, `DiffusionOps`, `OpBackend` (`op_ports`)
  - `MathOps` (+ the `impl_math_ops_via_core_ops!` macro that derives the
    `MathOps` wrapper from `CoreOps`)
  - `FusedOps`, `CollectiveOps`/`ShardedLoad`, `Sampler`
  - `LlmBackend` / `DiffusionBackend` / `Backend` (blanket-composed super-traits)
- `kv` — the paged KV-cache pool (`PagedKvPool`) and the `KvView`/`LayerKv` view
  types that appear in `FusedOps` signatures.
- `component` — `Component`/`Hidden`/`LayerRange`/`StageKind` model-structure types.

> History: the foundation was collapsed in from `infer-worker::domain` (it was
> mutually recursive at the item level — `Tensor` needs `MemoryPort`, `Storage`
> is the sole alloc caller, `ExecScope::enter` returns `ActiveGuard<Device>`,
> etc.). The interface half (`ports`/`kv`/`component`) was originally a separate
> `infer-backend-abi` crate; with no C-ABI boundary in the static design it was
> folded in here so backends depend on a single crate.

## ★ `infer-backend-cpu` — the CPU backend

A pure-Rust crate (deps: `infer-core`, `half`) implementing the op-port traits
for the `Cpu` device with scalar reference kernels (add, matmul, softmax,
rmsnorm, conv2d, groupnorm, attention, …). Carved out of the old
`infrastructure/cpu`. Contents: the `Cpu` device type, its `Device`/
`HostDevice`/`ExecDevice`/`MemoryPort`/`CoreOps`/`DiffusionOps`/`FusedOps`
impls, `CpuAllocator`, and `CpuTensorExt` (an extension trait providing the
Cpu-specific `zeros_cpu`/`from_slice` constructors, since `Tensor` lives in
`infer-core` and a crate may only add inherent methods to its own types).

It is **always linked** by the worker and is the baseline/terminal backend. It
implements `MemoryPort` and the exec vocabulary fully, a subset of the compute
ops natively, and returns `OpError::Unsupported` for the rest. Its ~16 op and
storage unit tests run via `cargo test -p infer-backend-cpu`.

## ★ `infer-backend-cuda` — the CUDA backend (toolchain isolated here)

The crate that confines the **entire CUDA toolchain** — nvcc, bindgen/libclang,
cuDNN, CUTLASS — to one leaf. Carved out of the old `infrastructure/cuda` tree.
Deps: `infer-core`, `half`, `tracing`; build-deps: `cc`, `bindgen`, `walkdir`.

Structure:
- `lib.rs` — the `Cuda` device, `CudaStream`, `CudaScope`, and the trait impls
  (`Device`, `ExecDevice`, `MemoryPort`, `MathOps`, `FusedOps`, `DiffusionOps`,
  `CollectiveOps`, …).
- `ffi` — bindgen-generated CUDA FFI bindings (`include!`'d from `OUT_DIR`).
- `config` (`CudaConfig`, `GraphSlot`), `error` (`CudaError`), `device_utils`.
- `kernels/` — 28 Rust kernel-wrapper modules over **29 `.cu` files** plus the
  vendored CUTLASS/CuTe headers under `kernels/third_party`.
- `build.rs` — the 471-line kernel build: walks `src/kernels/*.cu`, compiles them
  via `cc::Build::cuda(true)` into a static `infer_kernels` lib, runs `bindgen`
  over `src/wrapper.h`, discovers CUDA via conda/env heuristics, links
  cuBLAS/cuBLASLt/cuDNN/nvrtc.

The dtype hot path is unchanged: each op re-monomorphizes on the dtype
(`match T::DATA_TYPE → kernel_bf16x8 …`) inside the crate, reached by a direct
monomorphized call from the runtime. Builds standalone (`cargo build -p
infer-backend-cuda`, ~1m23s including kernels).

## ★ `infer-worker` — runtime + process host

The inference runtime and the thin ZMQ process host. Deps: `infer-protocol`,
`infer-core`, `infer-backend-cpu`, `infer-backend-cuda` (optional). **It no
longer has a `build.rs`** and no longer carries `cc`/`bindgen`/`walkdir` — the
CUDA toolchain moved entirely to `infer-backend-cuda`.

- `domain/` — now mostly a **re-export shim**: `pub use infer_core::{types, dtype,
  error, device, storage, tensor, exec, plan, ports, kv, component};` so all
  existing `crate::domain::…` paths resolve unchanged. It still owns the
  worker-local runtime types: `global_kv_alloc` (`GlobalKvAllocator`), `model`
  (`DecoderModel`/`Logits`/`ModelDims`), the runtime half of `plan`
  (`SeqStep`/`StepRequest`/`StepOutput`/…), and the relocated `kv_tests` /
  `tensor_tests`.
- `application/` — the runtime: `runtime` (`Runtime`/`DecodeEngine`), `serve_loop`,
  `decode_engine`, `worker_scheduler`, `kv_relief`, `decode_common` (the
  decode-heavy ones are `cuda`-gated), plus `hosting`, `sampler_stack`,
  `spec_runtime`, `tuning`, `worker_state`.
- `models/` — `decoder`, `llama3`, `qwen3`, `loader` (safetensors), `layers`, and
  `diffusion` (`cuda`-gated; Cuda-specific by design).
- `components/` — model building blocks: `attention`, `ffn_dense`, `ffn_moe`,
  `norm`, `linear`, `embed`, `lm_head`, `decoder_block`, `quant_linear`.
- `infrastructure/` — `pub use infer_backend_cpu as cpu;` (always) and
  `#[cfg(feature = "cuda")] pub use infer_backend_cuda as cuda;`, plus `io`
  (safetensors mmap) and `transport` (ZMQ pumps).
- `bin/worker_main.rs` — the `rustinfer-worker` binary (`required-features = ["cuda"]`).

Backend selection lives here: the `cuda` feature toggles whether
`infer-backend-cuda` is linked and which device the serve loop monomorphizes on.

## The other (pre-existing) crates

- **`infer-protocol`** — dependency-free wire types: the `scheduler↔worker` and
  `server↔scheduler` message envelopes, shared `common` types, and launch
  `config`. The only crate every process shares.
- **`infer-scheduler`** — the scheduler process (continuous-batching policy, KV
  orchestration, event loop). Depends only on `infer-protocol`.
- **`infer-server`** — the axum HTTP/SSE server (`api`, `router`, `chat`,
  `client` ZMQ, `middleware`, `state`). OpenAI-compatible. Depends only on
  `infer-protocol`.
- **`infer-frontend`** — a standalone Dioxus WASM UI (edition 2021, no
  in-workspace deps).

---

## Build matrix

| Command | Builds | Needs |
|---|---|---|
| `cargo check -p infer-worker --no-default-features` | core + CPU backend | **rustc only** — no nvcc, no build script |
| `cargo build -p infer-worker` (default = `cuda`) | core + CPU + CUDA | CUDA toolchain (in `infer-backend-cuda`) |
| `cargo build -p infer-backend-cuda` | the CUDA backend standalone | CUDA toolchain |
| `cargo test -p infer-backend-cpu` | CPU backend + its tests | rustc only |

Adding a new backend (e.g. ROCm/Metal): create `infer-backend-<name>` depending
on `infer-core`, implement the op-port traits for its device type, add it as an
optional dep of `infer-worker` behind a feature, and re-export it from
`infrastructure/mod.rs`. No change to `infer-core` or other backends.

## Why this shape

- **GPU-free core** — `infer-core` + `infer-backend-cpu` + the worker lib build
  with `rustc` alone; the CUDA toolchain is needed only to build one leaf crate.
- **Isolated kernels** — all `.cu`/CUTLASS/`build.rs` live in
  `infer-backend-cuda`; touching kernels never recompiles the core.
- **Zero runtime dispatch** — backends are selected at compile time and
  monomorphized; `Tensor<T, D>` and `match T::DATA_TYPE` stay direct calls. The
  decode hot path is byte-identical to the pre-refactor single-backend build.
- **Extensible** — a backend is one crate implementing one trait family.
