# RustInfer Worker — Architecture v2 (Framework Redesign)

> **Status: BINDING design document.** This governs the next phase of `crates/infer-worker`.
> It is **design-only**: trait/type/enum signatures + doc comments + how things compose.
> **No feature implementations**, no method bodies (default-impl logic appears only as a short `// ...` comment).
> Subsystem implementers build against the names and relationships fixed here.

---

## 1. Design goals & invariants

### Organizing principle

**A model is a sliceable stage list, driven over a narrow backend through one threaded execution scope.**

The worker has exactly **three** load-bearing axes of variation, and every future feature attaches along
precisely one of them:

- **BETWEEN stages** — the cut points `embed` → `decode_layers(range)` → `finalize`, with `Hidden`
  as a first-class *carried value*. → Pipeline parallelism, EAGLE / hidden-state taps, speculative
  draft+target co-hosting.
- **INSIDE a stage** — `Component`s: `Attention`, `DenseFfn | MoeFfn`, `Norm`, `Embed`, `LmHead`.
  → MoE (component swap), Tensor parallelism (collective at a fixed seam inside `Attention`/`Ffn`).
- **UNDER a stage** — the backend op floor + dtype/quant descriptors + `ExecScope`.
  → Heterogeneous backends, fp8/int4/mxfp4 quant, multi-stream / multi-device.

Every mandated axis is a geometric consequence of *where you cut*, *which component you instantiate*,
or *which floor impl you select* — never a reshape of a load-bearing signature.

### Hard invariants

These are the rules that make extension additive. They are checked against each other and hold as a set.

1. **Static dispatch is the perf spine.** `Tensor<T: Dtype, D: Device>` and all op methods are
   `Self`-typed **associated functions** (no `&self`), monomorphized per `(backend, dtype)`.
   **Two** `dyn` seams are permitted, both off the op path and both at the binary edge
   (see inv 11). No `dyn` on the op math path, ever.

2. **One foundational tensor bound.** `Device` *is* the memory-owning trait: it includes the
   `MemoryPort` surface. `Tensor<T, D: Device>` is the single tensor type, and **every** trait or
   struct that holds/produces a `Tensor` is bounded on `Device` (or a subtrait of it). There is no
   `Device` vs `MemoryPort` split at call sites — `MemoryPort` is folded into `Device`.

3. **Narrow required floor.** A new backend implements ONLY `MathOps` (~15 pure tensor→tensor ops).
   Every `FusedOps` / `DiffusionOps` method has a default body composed from `MathOps`, so correctness
   is free and a backend overrides only for speed.

4. **No orchestration physics in portable signatures.** `MathOps` methods take tensors + descriptors +
   a `&Scope` only — no `BatchPlan`, no paged-KV addressing, no device scratch in signatures.
   **`BatchPlan` is NOT generic over the backend** (`BatchPlan`, never `BatchPlan<D>`).

5. **One concurrency + device-activation seam — explicit `&Scope`, no thread-local.** All stream/queue
   access and `cudaSetDevice` go through `ExecScope`. The canonical scope type is
   `<D as Device>::Scope`. `ExecScope` is a **bound on that associated type**, never written
   `ExecScope<D>`. Entry is RAII (`enter()` activates the device for its lifetime). The scope is
   passed **explicitly** as `scope: &<D as Device>::Scope` to `MathOps`, and inside `&StepCtx` to fused
   ops / samplers. There is **no `active_scope()` thread-local and no `ScopeRef`** (both removed as
   unsound/ceremonial). The two current accessors (`tensor.device().config.stream` and the diffusion
   thread-local) are abolished in favor of this one explicit path.

6. **Receiver uniformity.** *All* capability-trait methods (`MathOps`, `FusedOps`, `DiffusionOps`,
   `CollectiveOps`) are associated functions with no `self` receiver, so `D::op(...)` works wherever
   only the type parameter `D` is in scope.

7. **`Hidden` is a carried value, not a workspace god-struct.** The model never owns a fixed dense-only
   view set. Cross-layer norm fusion is **un-fused at the abstraction level** (each layer owns its own
   norms); the speedup is re-won explicitly via the overridable `FusedOps::fused_add_rmsnorm` (and the
   other fused primitives), NEVER as a hidden cross-layer contract. Per-component scratch is private to
   each `Component`, sub-allocated from the single scope `Workspace`. There is **no central
   `ScratchRole` enum** (it would recreate the god-struct coupling).

8. **Open dtype + descriptor-driven quant.** `DataType` becomes an open `DTypeId` registry; `Dtype`
   keeps `SIZE_BYTES` + read/write. `QuantScheme {granularity, symmetry, packing, group}` replaces the
   bare `group_size`. fp8 e4m3/e5m2 are first-class `Dtype` impls now. New quant = a new `Packing`
   match-arm consumed inside one `matmul_quant` — never a new method. `bitcast` view replaces
   int4-as-i32.

9. **Mutation, not increment.** KV commit goes through `KvEdit` (`append(n)` / `truncate(to)` +
   per-seq `accepted_count`). The hardwired `seq.kv_len += 1` (decode_engine.rs:264) and one-slot-per-row
   alloc are abolished. The pool owns per-seq `kv_len` so `KvEdit` is self-contained.

10. **Orthogonal sidecars.** `CollectiveOps` (single-rank identity) is a backend capability;
    `Sampler` (greedy default) is a **strategy** object, **not** a backend capability — it is therefore
    NOT in the `Backend` alias. Single-rank/greedy paths stay byte-identical to today.

11. **GAT keystone + bounded dyn count.** `ExecScope` carries rank/stream/quant-tier as **fields**,
    never type parameters; `TopologyShape` (Copy rank/size data) is one such field. Communicator
    handles live on the backend (the device is its own `CollectiveOps`), reached via an associated fn —
    not as a generic field on the scope. The scope type never gains a second generic when
    TP/PP/multi-node/KV-tier arrive. The exactly-two permitted `dyn` seams are: **(a)** the
    architecture-erased model object (`Box<dyn DecoderModel<T, D>>` for a draft/aux model), and
    **(b)** the runtime sampler strategy (`Box<dyn Sampler<T, D>>`). Both are off the op path; both sit
    inside an already-monomorphized `(D, T)` entry.

12. **LLM path does not depend on the diffusion ceiling.** `DecoderModel`/`Runtime` are bounded on
    `LlmBackend` (`MathOps + FusedOps + CollectiveOps`), NOT on a god-alias that drags in `DiffusionOps`.

> **Must-override discipline (inv 3 corollary):** perf-critical `FusedOps` methods carry a
> `#[must_override(tier = "perf")]` marker. For any backend in the build's performance tier (today:
> `Cuda`) the marker forces a compile error unless an explicit override exists, so a correct-but-slow
> default can never silently ship on CUDA. Bring-up backends (`Cpu`, future `Metal`) are not in the
> tier and inherit defaults freely.

---

## 2. Module / crate tree

### Before (current)

```
infer-worker/src/
├── domain/ { ports/{device.rs, op_ports.rs}, model.rs, batch.rs (BatchPlan<D>), types.rs (closed DataType) }
├── application/ { model_runner/, decode_engine.rs (kv_len += 1), cuda_graph_runner.rs,
│                  forward_workspace.rs (dense-only view god-struct), serve_loop.rs, ... }
├── infrastructure/ { cuda/ (CudaConfig pub fields, thread-local stream), cpu/, transport/, io/ }
├── models/ { llama3.rs (cross-layer norm fusion @213-223), qwen3.rs, diffusion/ }
└── bin/worker_main.rs (ad-hoc `match model_type`, hardcodes <bf16, Cuda>)
```

### After (v2)

```
infer-worker/src/
├── domain/                          # pure, backend-agnostic
│   ├── tensor.rs                    # Tensor<T,D: Device> + bitcast/view (inv 2, 8)
│   ├── dtype/
│   │   ├── mod.rs                   # open dtype: trait Dtype, DTypeId registry; Fp8E4m3/Fp8E5m2 (inv 8)
│   │   └── quant.rs                 # QuantScheme {granularity,symmetry,packing,group} (inv 8)
│   ├── exec.rs                      # Device(+memory), ExecScope, Stream, ActiveGuard, StepCtx,
│   │                                #   TopologyShape, Rank, QuantTier, MaskHandle (inv 2,5,6,11)
│   ├── ports/
│   │   ├── math_ops.rs             # trait MathOps: Device — PORTABLE floor (~15 ops) (inv 3,4)
│   │   ├── fused_ops.rs            # trait FusedOps: MathOps — default-composed, overridable (inv 3,7)
│   │   ├── diffusion_ops.rs        # trait DiffusionOps: MathOps — family fused arms (inv 3,12)
│   │   ├── collective.rs           # trait CollectiveOps: Device — no-op single-rank (inv 6,10)
│   │   ├── sampler.rs              # trait Sampler<T,D> — probs + AcceptReject; NOT a backend cap (inv 10)
│   │   ├── backend.rs             # LlmBackend / DiffusionBackend / Backend blanket aliases (inv 12)
│   │   └── error.rs              # OpError::Unsupported{backend,op}; #[must_override] re-export
│   ├── component.rs               # trait Component, StageKind, Hidden, LayerRange, LayerWeights (inv 7)
│   ├── model.rs                   # trait DecoderModel: embed/decode_layers(range)/finalize (inv 7)
│   ├── kv.rs                      # PagedKvPool + KvView + KvEdit (inv 9)
│   └── plan.rs                    # BatchPlan (UN-parameterized), BatchKind, StepRequest, StepOutput (inv 4)
├── components/                     # concrete reusable stages, each owns its private scratch
│   ├── embed.rs · norm.rs · attention.rs · lm_head.rs · linear.rs · quant_linear.rs
│   ├── ffn_dense.rs               # DenseFfn: impl Component
│   ├── ffn_moe.rs                 # MoeFfn: Router + per-expert weights + grouped GEMM (MoE axis)
│   └── decoder_block.rs          # DecoderBlock<T,D,F: Component> — owns its norms (un-fused)
├── models/
│   ├── llama3.rs · qwen3.rs       # assemble stage lists; cross-layer norm UN-FUSED here
│   └── diffusion/                 # reuses DiffusionOps + ExecScope
├── application/
│   ├── runtime.rs                 # Runtime<T,D,M>: embed/decode_layers/finalize; one ExecScope
│   ├── hosting.rs                 # ModelHost: 1..N models in one runtime (spec, PP)
│   ├── exec_scope.rs            # RAII scope provisioning, fork()/record_event/wait_event (reserved)
│   ├── sampler_stack.rs        # GreedySampler / ChainSampler + verify; selected as Box<dyn Sampler>
│   ├── spec_runtime.rs         # draft+target co-hosting, linear chain (tree reserved)
│   ├── decode_engine.rs        # commit driven by StepOutput.accepted[seq] (inv 9)
│   ├── cuda_graph_runner.rs · serve_loop.rs · worker_scheduler.rs · kv_relief.rs · worker_state.rs
├── infrastructure/
│   ├── cuda/   # impl MathOps; override FusedOps/DiffusionOps hot arms; CudaScope; CollectiveOps (NCCL later)
│   ├── cpu/    # impl MathOps only — inherits FusedOps defaults; CollectiveOps no-op; no-op Scope
│   ├── <newbackend>/ # impl MathOps (~15) + Unsupported; own Scope (Metal cmd buf / Vulkan queue)
│   ├── transport/ · io/
└── bin/worker_main.rs            # macro-generated match over SHIPPED (backend×dtype×arch) tuples
```

---

## 3. Core abstractions

All sketches below are mutually consistent: same generic parameters, the **same scope-threading rule**
(MathOps takes `scope: &<Self as Device>::Scope`; fused ops + sampler take `ctx: &StepCtx<'_, Self>`
which carries the scope + plan + mask), the same `Tensor<T, D: Device>` bound, and uniform
no-`self` associated-fn receivers on every capability trait.

### 3.1 Execution & device seam — `domain/exec.rs`

```rust
use std::fmt::Debug;
use std::ptr::NonNull;
use crate::domain::ports::error::OpResult;

/// Stable identity of a physical device within this process (CUDA ordinal; 0 for CPU).
/// Allocation is keyed on this; the active guard drives `cudaSetDevice` on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId(pub i32);

/// A compute target (Cuda / Cpu / future ROCm / Metal / Vulkan) AND its memory surface.
///
/// `Device` is the single foundational bound (inv 2): it folds in the former `MemoryPort`,
/// so `Tensor<T, D: Device>` and every tensor-holding type bound on `Device` compose without a
/// second memory trait. All capability traits are `: Device`.
pub trait Device: Clone + Send + Sync + Debug + 'static {
    /// The concrete execution scope type for this device. `ExecScope` is a BOUND on it,
    /// never written `ExecScope<Self>` (inv 5). CUDA: stream + handles + workspace + rank.
    /// CPU: a ZST no-op scope. Metal: a command buffer. Vulkan: a queue + pool.
    type Scope: ExecScope<Device = Self>;

    fn device_id(&self) -> DeviceId;
    fn name(&self) -> &'static str;

    // ── Memory surface (folded in from the old MemoryPort) ──
    /// Allocate `size` zeroed bytes on this device (device-keyed, scope-independent).
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>>;
    /// # Safety: `ptr`/`size` must match a prior live `alloc_bytes`.
    unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize);
    /// Synchronous H2D copy on `scope`'s stream, then sync that stream.
    /// # Safety: device/host ptrs ≥ `size`.
    unsafe fn upload(&self, scope: &Self::Scope, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;
    /// Async H2D copy on `scope`'s stream; no sync (graph-capture safe). # Safety: as `upload`.
    unsafe fn upload_async(&self, scope: &Self::Scope, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;
    /// Synchronous D2H copy on `scope`'s stream. # Safety: ptrs ≥ `size`.
    unsafe fn download(&self, scope: &Self::Scope, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()>;
    /// On-device disjoint copy on `scope`'s stream. # Safety: disjoint device ptrs ≥ `size`.
    unsafe fn copy_device_to_device(&self, scope: &Self::Scope, dst: NonNull<u8>, src: NonNull<u8>, size: usize) -> OpResult<()>;
}

/// Marker: host-addressable memory (enables `Tensor::as_slice`). CPU only.
pub trait HostDevice: Device {}

/// The unified execution-context value (inv 5, 11). Holds device handle + current stream + GEMM
/// workspace + this scope's `Rank` + active `QuantTier`, ALL AS FIELDS (never type params).
/// Replaces leaked `CudaConfig` pub fields AND the diffusion thread-local. Passed EXPLICITLY.
pub trait ExecScope: Send + Sync + Sized + 'static {
    type Device: Device<Scope = Self>;
    /// Backend-native ordered lane (CUDA stream / Metal cmd buffer / Vulkan queue / CPU `()`).
    type Stream: Stream;

    /// Activate this scope's device for the guard lifetime (RAII `cudaSetDevice` + restore on drop).
    /// The ONLY place `cudaSetDevice` is called on the hot path → multi-device correct by construction
    /// (inv 5). `!Send` guard: a scope is active on exactly one thread for its lifetime.
    fn enter(&self) -> ActiveGuard<'_, Self::Device>;

    /// The single stream accessor (replaces both legacy accessors).
    fn stream(&self) -> &Self::Stream;
    /// This scope's mesh position (Copy field). `CollectiveOps` reads it; `Rank::SINGLE` = no-op path.
    fn rank(&self) -> Rank;
    /// Mesh shape (rank/size per axis), a Copy field (inv 11). No generic on the scope.
    fn topology(&self) -> TopologyShape;
    /// Active KV/weight quant tier (reserved). Defaults to `QuantTier::None`.
    fn quant_tier(&self) -> QuantTier;
    /// Scratch GEMM/reduction workspace owned by this scope (replaces leaked `CudaConfig::workspace`).
    fn workspace(&self) -> &Workspace<Self::Device>;
    /// Block host until this scope's stream drains. CPU: no-op.
    fn synchronize(&self) -> OpResult<()>;

    // ── Reserved multi-stream / cross-stage seam (single-stream ships today; see §6) ──
    // type Event: Event;
    // fn fork(&self) -> OpResult<Self>;
    // fn record_event(&self) -> OpResult<Self::Event>;
    // fn wait_event(&self, ev: &Self::Event) -> OpResult<()>;
}

/// Backend-native ordered lane marker (op code never names the concrete type).
pub trait Stream: Send + Sync + 'static {}

/// RAII guard from `ExecScope::enter()`. While alive the scope's device is current. `!Send` on purpose.
pub struct ActiveGuard<'a, D: Device> {
    _scope: &'a D::Scope,
    _prev_device: DeviceId,
    _not_send: core::marker::PhantomData<*const ()>,
}
// Drop: restore `cudaSetDevice(_prev_device)`.

/// Scope-owned scratch arena (replaces leaked `CudaConfig::workspace`). Components sub-allocate
/// their private scratch from here (inv 7); nothing constructs raw GEMM scratch by hand.
pub struct Workspace<D: Device> { _ptr: Option<NonNull<u8>>, _size: usize, _d: core::marker::PhantomData<D> }

/// The per-STEP carrier threaded between stages (not between ops). Holds the scope + the
/// un-parameterized `&BatchPlan`. Mutable execution state lives behind the scope, so `StepCtx`
/// never becomes mutable or gains a parameter when TP/PP/spec arrive. The tree/Medusa mask is
/// carried as ONE reserved field (`MaskHandle`), not as a raw pointer (inv 11; see §6).
pub struct StepCtx<'a, D: Device> {
    scope: &'a D::Scope,
    plan: &'a crate::domain::plan::BatchPlan,
    _marker: core::marker::PhantomData<D>,
}

impl<'a, D: Device> StepCtx<'a, D> {
    pub fn new(scope: &'a D::Scope, plan: &'a crate::domain::plan::BatchPlan) -> Self;
    /// The active scope (fused ops call `ctx.scope().enter()` once, then pass `ctx.scope()` to MathOps).
    pub fn scope(&self) -> &D::Scope;
    /// Pure, backend-agnostic batch metadata (inv 4: never `BatchPlan<D>`).
    pub fn plan(&self) -> &crate::domain::plan::BatchPlan;
}

/// A scope's position in the mesh — a Copy FIELD on the scope (inv 11), read by collectives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rank {
    pub tp_rank: usize, pub pp_rank: usize, pub dp_rank: usize, pub node_rank: usize, pub world_rank: usize,
}
impl Rank { pub const SINGLE: Rank = Rank { tp_rank: 0, pp_rank: 0, dp_rank: 0, node_rank: 0, world_rank: 0 }; }

/// (rank, size) for one parallel axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RankPair { pub rank: usize, pub size: usize }

/// Mesh shape: ranks/sizes only (Copy). The SINGLE source carried on the scope (inv 11). Communicator
/// handles do NOT live here — they sit on the backend's `CollectiveOps` impl, reached via an
/// associated fn, so the scope never gains a `<C>` generic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TopologyShape { pub tp: RankPair, pub pp: RankPair, pub dp: RankPair, pub node: RankPair }
impl TopologyShape {
    pub const SINGLE: TopologyShape = TopologyShape {
        tp: RankPair { rank: 0, size: 1 }, pp: RankPair { rank: 0, size: 1 },
        dp: RankPair { rank: 0, size: 1 }, node: RankPair { rank: 0, size: 1 },
    };
    /// Built from the reserved protocol `LoadModel{tp_rank,tp_size,pp_rank,pp_size}` fields.
    pub fn from_load_model(load: &infer_protocol::scheduler_to_worker_control::LoadModel) -> Self;
    pub fn world_size(&self) -> usize; // tp.size * pp.size * dp.size * node.size
    pub fn rank_in(&self, axis: CommAxis) -> usize;
    pub fn group_size(&self, axis: CommAxis) -> usize;
    pub fn is_pp_first(&self) -> bool;
    pub fn is_pp_last(&self) -> bool;
}

/// RESERVED quant tier carried by the scope. `None` today; new tiers are new arms, never new methods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantTier { None /*, Fp8Kv, Int4Kv (reserved) */ }

/// Opaque backend-agnostic handle to a device-resident attention mask (tree/Medusa, reserved).
/// A registry index, NOT a raw pointer — resolved through the scope at the kernel site (§6).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskHandle(pub u64);

/// Re-export of the `must_override` attribute proc-macro (crate `infer-worker-macros`).
pub use infer_worker_macros::must_override;
```

> The `CommAxis` enum referenced above is defined in §3.3 (collectives) and re-used here.

### 3.2 Op ports — `domain/ports/{error,math_ops,fused_ops,diffusion_ops,backend}.rs`

```rust
// ── error.rs ──
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error("shape error: {0}")]                                  Shape(String),
    #[error("not contiguous")]                                    NotContiguous,
    /// The sanctioned bring-up valve: a backend declines an op it has not kerneled.
    /// Static strings → alloc-free + greppable.
    #[error("unsupported op '{op}' on backend '{backend}'")]      Unsupported { backend: &'static str, op: &'static str },
    #[error("kernel failed: {0}")]                                Kernel(String),
    #[error("shutdown requested")]                                Shutdown,
}
impl OpError { pub fn unsupported(backend: &'static str, op: &'static str) -> Self; }
pub type OpResult<T> = std::result::Result<T, OpError>;
```

```rust
// ── math_ops.rs — THE PORTABLE FLOOR (inv 3, 4, 6) ──
use crate::domain::exec::{Device, ExecScope};
use crate::domain::dtype::{Dtype, quant::QuantScheme};
use crate::domain::tensor::{Tensor, Shape};
use crate::domain::ports::error::OpResult;

/// The ONLY surface a new backend is REQUIRED to implement. Every method is dtype-generic, pure
/// tensor-in/tensor-out, and portable. The active stream/device is reached through the EXPLICIT
/// `scope: &<Self as Device>::Scope` (inv 5) — there is no thread-local. No `BatchPlan`, no paged-KV
/// addressing, no device scratch in any signature (inv 4). All methods are associated fns (inv 6).
pub trait MathOps: Device {
    /// Allocate a contiguous zeroed tensor. // default: Tensor::<T,Self>::zeros(shape, device)
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>>;

    // ── element-wise ──
    fn add<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    fn add_inplace<T: Dtype>(scope: &Self::Scope, dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()>;
    fn ewise_mul<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    fn scalar_mul_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()>;
    fn broadcast_mul_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, scale: &Tensor<T, Self>) -> OpResult<()>;
    fn broadcast_add_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, bias: &Tensor<T, Self>) -> OpResult<()>;

    // ── linear algebra ──
    /// Dense matmul `[M,K] × [N,K]^T → [M,N]`, same dtype.
    fn matmul<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Quantized matmul. `scheme` (inv 8) fully describes granularity/symmetry/packing/group — it
    /// REPLACES the bare `group_size`. `zeros` is `Some` iff `scheme.symmetry == Asymmetric`. A backend
    /// matches on `scheme.packing` and returns `Unsupported` for un-kerneled packings. New quant family
    /// = new `Packing` arm consumed HERE, never a new method.
    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<A, Self>, weight: &Tensor<W, Self>, output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>, zeros: Option<&Tensor<W, Self>>, scheme: &QuantScheme,
    ) -> OpResult<()>;

    // ── normalization (portable primitive — un-fused, inv 7) ──
    fn rmsnorm<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>, eps: f32) -> OpResult<()>;
    fn rmsnorm_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32) -> OpResult<()>;

    // ── activations ──
    fn silu_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>) -> OpResult<()>;
    fn softmax<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

    // ── RoPE (portable; paged scatter is NOT here) ──
    fn rope_inplace<T: Dtype>(
        scope: &Self::Scope, q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>, cos: &Tensor<T, Self>, positions: &Tensor<i32, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize,
    ) -> OpResult<()>;

    // ── dense attention primitive (NO KV pool, NO plan) ──
    /// SDPA over CONTIGUOUS materialized q/k/v with an OPTIONAL additive mask. The portable attention
    /// floor: `FusedOps::attention_paged` default-composes it by gathering paged K/V first. GQA-aware.
    fn sdpa<T: Dtype>(
        scope: &Self::Scope, q: &Tensor<T, Self>, k: &Tensor<T, Self>, v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>, mask: Option<&Tensor<T, Self>>,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()>;

    // ── embedding / shape / dtype ──
    fn embedding<T: Dtype>(scope: &Self::Scope, table: &Tensor<T, Self>, indices: &Tensor<i32, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;
    fn split_cols<T: Dtype>(scope: &Self::Scope, src: &Tensor<T, Self>, dst: &mut Tensor<T, Self>, rows: usize, total_cols: usize, col_offset: usize, dst_cols: usize) -> OpResult<()>;
    fn concat_seq<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Numeric cast between dtypes on the same device. Spans fp8 ↔ bf16/f16/f32 (fp8 are ordinary Dtypes).
    fn cast<S: Dtype, T: Dtype>(scope: &Self::Scope, src: &Tensor<S, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Reinterpret bytes WITHOUT copy (inv 8): `Tensor<T>` view over the same storage as `Tensor<S>`.
    /// REPLACES int4-as-i32. Errors unless `S::SIZE_BYTES * src.numel()` divides `T::SIZE_BYTES`.
    /// Pure view math (O(1) Arc bump) — portable default, never overridden.
    fn bitcast<S: Dtype, T: Dtype>(src: &Tensor<S, Self>, new_shape: Shape) -> OpResult<Tensor<T, Self>>;
}
```

```rust
// ── fused_ops.rs — THE LLM CEILING (inv 3, 4, 7) ──
use crate::domain::exec::StepCtx;
use crate::domain::kv::{KvView, LayerKv};
use crate::domain::plan::BatchPlan;

/// Orchestration / fused LLM ops. EVERY method has a default body composed purely from `MathOps`
/// (inv 3): a backend implementing only `MathOps` satisfies `FusedOps` and runs LLMs correctly
/// (slowly). CUDA overrides hot arms. Orchestration physics (paged KV, the un-parameterized
/// `&BatchPlan`, the mask mode) live HERE, reached via `ctx: &StepCtx` (which carries scope + plan).
/// All methods associated fns (inv 6).
pub trait FusedOps: MathOps {
    /// Fused `residual += input; output = rmsnorm(residual, weight, eps)`.
    /// This is where the OLD cross-layer norm fusion is re-won EXPLICITLY (inv 7).
    /// default: add_inplace(residual,input); rmsnorm(residual,weight,output,eps)
    fn fused_add_rmsnorm<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        output: &mut Tensor<T, Self>, residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32,
    ) -> OpResult<()>;

    /// Packed SwiGLU `gate_up[rows,2*inter] → out[rows,inter]`.
    /// default: split_cols gate&up; silu_inplace(gate); ewise_mul(gate,up,out)
    fn swiglu_packed<T: Dtype>(ctx: &StepCtx<'_, Self>, gate_up: &Tensor<T, Self>, out: &mut Tensor<T, Self>, rows: usize, inter: usize) -> OpResult<()>;

    /// Split fused `[num_tokens, qkv_dim]` into Q/K/V. default: three split_cols.
    fn split_qkv<T: Dtype>(ctx: &StepCtx<'_, Self>, qkv: &Tensor<T, Self>, q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>, v: &mut Tensor<T, Self>, num_tokens: usize, q_dim: usize, kv_dim: usize) -> OpResult<()>;

    /// Paged attention over a layer's KV view. Mask discipline comes from `plan.kind`'s `MaskMode`
    /// (NOT a bare `is_causal: i32`); scratch comes from `ctx.scope().workspace()`. `kv` bundles the
    /// pool slice + device index tensors so `BatchPlan` stays un-parameterized (inv 4).
    /// default: gather paged K/V → contiguous, build additive mask from plan, call MathOps::sdpa.
    #[must_override(tier = "perf")]
    fn attention_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &Tensor<T, Self>, kv: &KvView<'_, T, Self>, output: &mut Tensor<T, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()>;

    /// Scatter K/V rows into a layer's paged pool. Index tensors come from `kv.index`; signature is
    /// paging-geometry only. Positionally overwrite-safe. The `LayerKv` write handle is `&mut`.
    /// default: per-token block-id arithmetic + Device::copy_device_to_device.
    #[must_override(tier = "perf")]
    fn scatter_kv_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        k_src: &Tensor<T, Self>, v_src: &Tensor<T, Self>,
        layer: &mut LayerKv<'_, T, Self>, kv_dim: usize,
    ) -> OpResult<()>;

    /// Fused Q/K-RMSNorm + RoPE + paged scatter (Qwen path). The canonical proof fused = composed.
    /// default: optional rmsnorm_inplace on Q/K head-views; rope_inplace; scatter_kv_paged.
    #[must_override(tier = "perf")]
    fn qkv_norm_rope_scatter<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>, v: &Tensor<T, Self>,
        q_weight: Option<&Tensor<T, Self>>, k_weight: Option<&Tensor<T, Self>>, q_eps: f32, k_eps: f32,
        sin: &Tensor<T, Self>, cos: &Tensor<T, Self>, positions: &Tensor<i32, Self>,
        layer: &mut LayerKv<'_, T, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize, kv_dim: usize,
    ) -> OpResult<()>;

    /// MoE grouped expert GEMM. `expert_offsets[num_experts+1]` defines token groups (post sort by
    /// expert); `weights` is the stacked expert tensor; optional `scheme` enables quantized experts.
    /// default: loop experts, slice token rows by offset, matmul/matmul_quant each group (dropless,
    /// single-GPU). Expert-parallel later adds CollectiveOps::all_to_all in `MoeFfn`, NOT here.
    #[must_override(tier = "perf")]
    fn grouped_expert_gemm<A: Dtype, W: Dtype, O: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<A, Self>, weights: &Tensor<W, Self>, output: &mut Tensor<O, Self>,
        expert_offsets: &Tensor<i32, Self>,
        scales: Option<&Tensor<A, Self>>, zeros: Option<&Tensor<W, Self>>, scheme: Option<&QuantScheme>,
    ) -> OpResult<()>;
}
```

> **No `fused_decode_layer` in the ceiling.** Per critique, a whole-layer megakernel with `&dyn
> LayerWeights` would put `dyn` on the op path (violates inv 1) and is not additive for MoE. The
> cross-layer fusion is recovered by `fused_add_rmsnorm` + `qkv_norm_rope_scatter` + `swiglu_packed`.
> If a true whole-layer megakernel is later justified, it is a **concrete** method on the concrete
> `DecoderBlock<F>` (static, knows its `F`), not a dyn-weights trait method.

```rust
// ── diffusion_ops.rs — FAMILY CEILING (inv 3, 12) ──
/// Diffusion-family fused arms, same floor/ceiling discipline (every method default-composes from
/// MathOps). NOT in the LLM `Backend` bound (inv 12). An LLM-only backend leaves these `Unsupported`.
pub trait DiffusionOps: MathOps {
    fn conv2d<T: Dtype>(ctx: &StepCtx<'_, Self>, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: Option<&Tensor<T, Self>>, output: &mut Tensor<T, Self>, stride: usize, padding: usize) -> OpResult<()>;
    fn groupnorm<T: Dtype>(ctx: &StepCtx<'_, Self>, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>, output: &mut Tensor<T, Self>, num_groups: usize, eps: f32) -> OpResult<()>;
    /// default: groupnorm then silu_inplace.
    fn groupnorm_silu<T: Dtype>(ctx: &StepCtx<'_, Self>, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>, output: &mut Tensor<T, Self>, num_groups: usize, eps: f32) -> OpResult<()>;
    fn layernorm<T: Dtype>(ctx: &StepCtx<'_, Self>, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>, output: &mut Tensor<T, Self>, eps: f32) -> OpResult<()>;
    fn upsample_nearest_2x<T: Dtype>(ctx: &StepCtx<'_, Self>, input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;
    fn apply_rope_interleaved<T: Dtype>(ctx: &StepCtx<'_, Self>, x: &mut Tensor<T, Self>, cos: &Tensor<f32, Self>, sin: &Tensor<f32, Self>, head_dim: usize) -> OpResult<()>;
    fn tanh_inplace<T: Dtype>(ctx: &StepCtx<'_, Self>, x: &mut Tensor<T, Self>) -> OpResult<()>;
    // NOTE: the legacy diffusion `sdpa`/`silu_inplace_diff` are GONE — folded into the unified
    // `MathOps::sdpa(mask: Option)` and `MathOps::silu_inplace`.
}
```

```rust
// ── backend.rs — capability aliases (inv 10, 12) ──
/// The clean bound the DECODER path carries. Excludes DiffusionOps (inv 12) and Sampler (inv 10:
/// Sampler is a strategy, not a backend capability). Blanket-impl'd, zero per-backend boilerplate.
pub trait LlmBackend: FusedOps + CollectiveOps {}
impl<D: FusedOps + CollectiveOps> LlmBackend for D {}

/// The bound a DIFFUSION model carries.
pub trait DiffusionBackend: DiffusionOps + CollectiveOps {}
impl<D: DiffusionOps + CollectiveOps> DiffusionBackend for D {}

/// Convenience super-alias for a backend that does everything (e.g. a unified CUDA build).
pub trait Backend: LlmBackend + DiffusionBackend {}
impl<D: LlmBackend + DiffusionBackend> Backend for D {}
```

### 3.3 Collectives & topology — `domain/ports/collective.rs`

```rust
use crate::domain::exec::{Device, ExecScope};
use crate::domain::dtype::Dtype;
use crate::domain::tensor::Tensor;
use crate::domain::ports::error::OpResult;

/// Which logical mesh dimension a collective runs over. Closed on purpose: every parallelism axis the
/// worker will host is one of these. A new axis = one variant — never a new trait method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum CommAxis { Tp, Pp, Dp, Ep }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum ReduceOp { Sum, Max, Min, Avg }

/// Communication capability over the process mesh. Sidecar `: Device` (shares the tensor spine) but
/// NOT part of the portable math floor. All methods are associated fns (inv 6) taking the explicit
/// `scope` (rank is read from `scope.rank()`/`scope.topology()`, never an argument — inv 11).
/// Communicator handles are reached via `comm(axis)` on the concrete backend; they are NOT a generic
/// field on the scope, so inv 11 (no second scope generic) holds.
///
/// Single-rank impl makes every method a no-op identity (all_reduce returns input unchanged,
/// all_gather copies the local shard, send/recv are unreachable) → today's path is byte-identical.
pub trait CollectiveOps: Device {
    /// One process group handle (CUDA: `ncclComm_t` newtype; CPU/single-rank: a ZST).
    type Comm: Send + Sync;

    /// Borrow this backend's communicator for `axis` (None when the axis is size-1). Concrete-typed,
    /// never on an op signature.
    fn comm(scope: &Self::Scope, axis: CommAxis) -> Option<&Self::Comm>;

    /// In-place all-reduce across `axis` (the TP work-horse at the row-parallel seam).
    fn all_reduce<T: Dtype>(scope: &Self::Scope, axis: CommAxis, op: ReduceOp, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Gather each rank's `shard` into `out` along `dim` on every rank.
    fn all_gather<T: Dtype>(scope: &Self::Scope, axis: CommAxis, dim: usize, shard: &Tensor<T, Self>, out: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Reduce + scatter so each rank keeps its `1/group_size` slice.
    fn reduce_scatter<T: Dtype>(scope: &Self::Scope, axis: CommAxis, op: ReduceOp, dim: usize, buf: &Tensor<T, Self>, out: &mut Tensor<T, Self>) -> OpResult<()>;
    /// Broadcast `buf` from `root` (index within `axis`) to all members.
    fn broadcast<T: Dtype>(scope: &Self::Scope, axis: CommAxis, root: usize, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// P2P send/recv of `Hidden` across PP stages (`peer` is an index within `axis`).
    fn send<T: Dtype>(scope: &Self::Scope, axis: CommAxis, peer: usize, buf: &Tensor<T, Self>) -> OpResult<()>;
    fn recv<T: Dtype>(scope: &Self::Scope, axis: CommAxis, peer: usize, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// All-to-all exchange. RESERVED for expert-parallel MoE dispatch/combine (§6).
    fn all_to_all<T: Dtype>(scope: &Self::Scope, axis: CommAxis, send_chunks: &[Tensor<T, Self>], recv_chunks: &mut [Tensor<T, Self>]) -> OpResult<()>;
    /// Mesh barrier on `axis` (debug/drain). Single-rank: no-op.
    fn barrier(scope: &Self::Scope, axis: CommAxis) -> OpResult<()>;
}

/// How one weight tensor is partitioned across the TP group (a LOAD-TIME concern; no runtime
/// signature changes for TP). New partition styles = new variant, never a new loader method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardSpec {
    Replicated,
    ColumnParallel { dim: usize },  // q/k/v proj, gate/up
    RowParallel    { dim: usize },  // o_proj, down_proj → all_reduce(Sum) at the seam
    VocabParallel  { dim: usize },  // embedding / lm_head (reserved gather at sampling)
}

/// Load-time slice hook: a Component pulls only its own TP shard. `tp_size == 1` always returns the
/// full tensor regardless of `spec`, so existing models load byte-identically.
pub trait ShardedLoad: Device {
    fn load_shard<T: Dtype>(&self, rank: crate::domain::exec::Rank, name: &str, spec: ShardSpec) -> OpResult<Tensor<T, Self>>;
}
```

### 3.4 Model & layers — `domain/{component,model}.rs`, `components/*`

```rust
// ── component.rs ──
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::{backend::LlmBackend, error::OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::dtype::Dtype;

/// First-class inter-stage carried value (inv 7). Replaces the dense-only `LlmForwardWorkspace`.
/// Carries ONLY the residual stream `[num_tokens, dim]`. There is NO central scratch role enum — each
/// `Component` owns its private scratch (sub-allocated from the scope `Workspace`). Address stability
/// for CUDA-graph capture is provided by the Runtime allocating the `Hidden` slot ONCE and passing it
/// `&mut` (see §3.8); components never allocate it per step.
pub struct Hidden<T: Dtype, D: LlmBackend> {
    /// Residual stream `[num_tokens, dim]`: the single value every stage reads/writes.
    pub stream: Tensor<T, D>,
}
impl<T: Dtype, D: LlmBackend> Hidden<T, D> {
    pub fn num_tokens(&self) -> usize;
    /// EAGLE / spec / PP hidden-state tap between two stages (shallow by default).
    pub fn tap_stream(&self, deep: bool) -> OpResult<Tensor<T, D>>;
}

/// What kind of stage a component is — drives PP cut decisions + tap introspection.
/// `#[non_exhaustive]`: a future kind (cross-attn) is one new arm.
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StageKind { Embed, Norm, Attention, Ffn /* dense or MoE — swap is invisible */, LmHead, DecoderBlock }

/// Inclusive-exclusive decoder-layer range for sliced forward (PP / EAGLE / spec).
#[derive(Clone, Copy, Debug)]
pub struct LayerRange { pub start: usize, pub end: usize }
impl LayerRange {
    pub fn all(num_layers: usize) -> Self;
    pub fn single(i: usize) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    /// Even split of `num_layers` across `pp_size` → this `pp_rank`'s band (the only PP-specific calc).
    pub fn for_pp_rank(pp_rank: usize, pp_size: usize, num_layers: usize) -> Self;
}

/// The atomic compute brick. Maps `Hidden -> Hidden` over backend `D`, given the per-step `StepCtx`
/// (which carries the scope + plan). `&self` (inference is pure). TP collective seams live at FIXED
/// points INSIDE `Attention`/`Ffn` impls — not in this signature. `kv` is `Some` only for
/// attention-bearing stages. The view is a single `KvView` whose mutable layer access is borrow-split
/// internally for scatter-then-attend (inv 9; see §3.5).
pub trait Component<T: Dtype, D: LlmBackend> {
    fn kind(&self) -> StageKind;
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
}
```

```rust
// ── model.rs ──
use crate::domain::component::{Hidden, LayerRange, StageKind};

/// Geometric description. WIDENED with MoE dims defaulting to 0 so dense models stay byte-identical.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub dim: usize, pub q_dim: usize, pub kv_dim: usize, pub qkv_dim: usize,
    pub intermediate_size: usize, pub vocab_size: usize,
    pub head_num: usize, pub head_dim: usize, pub kv_head_num: usize, pub num_layers: usize,
    // ── MoE widening: ALL default 0 for dense (additive) ──
    pub num_experts: usize, pub experts_per_tok: usize, pub moe_intermediate_size: usize, pub num_shared_experts: usize,
}
impl Default for ModelDims { fn default() -> Self; }
impl ModelDims {
    pub fn validate(&self) -> OpResult<()>;
    pub fn is_moe(&self) -> bool; // num_experts > 0
}

/// Logits returned by `finalize`: `[num_sampled_rows, vocab]`.
pub struct Logits<T: Dtype, D: LlmBackend>(pub Tensor<T, D>);

/// Which token rows `finalize` projects (avoids the implicit "all rows" assumption).
pub enum SampleRows<'a> { All, LastPerSeq, Explicit(&'a [i32]) }

/// The sliceable decoder model. NO monolithic kernel-fused contract. `forward` exists ONLY as a
/// provided default composing the three seams, so existing call sites keep a one-shot path while
/// PP/EAGLE/spec use the slices directly. Bound on `LlmBackend` (inv 12), the single clean bound.
pub trait DecoderModel<T: Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    /// Stage descriptors in execution order (for the PP planner; no execution).
    fn stages(&self) -> &[StageKind];

    /// SEAM 1 — token ids → initial `Hidden`. Writes INTO the runtime-provisioned, address-stable
    /// `hidden` slot (does NOT allocate — preserves CUDA-graph pointer stability).
    fn embed(&self, input_ids: &Tensor<i32, D>, hidden: &mut Hidden<T, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;

    /// SEAM 2 — run layers `[range.start, range.end)` in place. THE PP/EAGLE/spec slice. Cross-layer
    /// norm is UN-FUSED: each block applies its OWN input+post-attn norms; fusion is re-won only
    /// inside `FusedOps::fused_add_rmsnorm`, never as a model-level contract.
    fn decode_layers(&self, range: LayerRange, hidden: &mut Hidden<T, D>, kv: &mut KvView<'_, T, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;

    /// SEAM 3 — final norm + LM head → logits (last PP stage only). `rows` selects projected tokens.
    fn finalize(&self, hidden: &Hidden<T, D>, rows: SampleRows<'_>, ctx: &StepCtx<'_, D>) -> OpResult<Logits<T, D>>;

    /// Provided MONOLITHIC default = embed → decode_layers(all) → finalize. Single-rank dense path is
    /// byte-identical to today's `forward`.
    fn forward(&self, input_ids: &Tensor<i32, D>, hidden: &mut Hidden<T, D>, kv: &mut KvView<'_, T, D>, rows: SampleRows<'_>, ctx: &StepCtx<'_, D>) -> OpResult<Logits<T, D>>;
}
```

```rust
// ── components/ffn_dense.rs & ffn_moe.rs — dense↔MoE is a COMPONENT SWAP ──
use crate::components::linear::Linear;

/// Dense SwiGLU FFN (current Llama3/Qwen3 path). Owns its post-attention norm (cross-layer fusion
/// gone). Private scratch sub-allocated from the scope workspace.
pub struct DenseFfn<T: Dtype, D: LlmBackend> { pub gate_up_proj: Linear<T, D>, pub down_proj: Linear<T, D> }
impl<T: Dtype, D: LlmBackend> Component<T, D> for DenseFfn<T, D> {
    fn kind(&self) -> StageKind; // Ffn
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // gate_up = gate_up_proj(h); swiglu_packed; down_proj
}

/// Sparse MoE FFN. SAME `Component` contract → swapping it in is the ENTIRE MoE model change.
/// Router top-k + grouped_expert_gemm are FusedOps (MathOps-composed defaults). Expert intermediates
/// live in this component's PRIVATE scratch — no shared role enum, no workspace reshape.
pub struct MoeFfn<T: Dtype, D: LlmBackend> {
    pub router: Linear<T, D>,
    pub expert_gate_up: Tensor<T, D>, // [num_experts, 2*moe_inter, dim]
    pub expert_down: Tensor<T, D>,    // [num_experts, dim, moe_inter]
    pub shared: Option<DenseFfn<T, D>>,
    pub experts_per_tok: usize,
}
impl<T: Dtype, D: LlmBackend> Component<T, D> for MoeFfn<T, D> {
    fn kind(&self) -> StageKind; // Ffn (swap invisible to runtime)
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // logits=router(h); (ids,w)=topk; permute by expert into private scratch; grouped_expert_gemm;
    // unpermute+weight; + shared.run() if present. Dropless, fixed per-expert capacity (graph-able, §6).
}

// ── components/decoder_block.rs — generic over its FFN; norms OWNED here (un-fused) ──
use crate::components::{norm::RmsNorm, attention::Attention};
pub struct DecoderBlock<T: Dtype, D: LlmBackend, F: Component<T, D>> {
    pub input_layernorm: RmsNorm<T, D>,
    pub attention: Attention<T, D>,
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub ffn: F, // DenseFfn or MoeFfn — chosen at model-build time, static dispatch
}
impl<T: Dtype, D: LlmBackend, F: Component<T, D>> Component<T, D> for DecoderBlock<T, D, F> {
    fn kind(&self) -> StageKind; // DecoderBlock (the PP cut unit)
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // input_layernorm → attention(scatter then attend on one KvView) → fused_add_rmsnorm → ffn.run
}
```

### 3.5 KV cache & batch plan — `domain/{kv,plan}.rs`

```rust
// ── plan.rs — BatchPlan UN-parameterized (inv 4). SINGLE authoritative definition. ──
use crate::domain::exec::MaskHandle;

/// Q-tile size for the ragged paged-attention kernel (must match `kBlockM`).
pub const RAGGED_Q_TILE: i32 = 128;

/// Attention mask discipline — the typed replacement for the bare `is_causal: i32`. ONE enum; spec
/// reuses it (no parallel `SpecMaskMode`). New regime = new arm + a kernel branch, never a new method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    Full,                              // diffusion / encoder self-attention
    Causal,                           // standard autoregressive (today's is_causal==1)
    SlidingWindow { window: u32 },    // RESERVED
    Tree,                             // RESERVED tree/Medusa — reads the BatchKind::Spec mask handle
}

/// What flavour of attention this step runs. Drives kernel dispatch + the mask the attention
/// component builds. `Spec` carries the reserved tree mask handle (ONE place; not also on StepCtx).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    DecodeOnly,                                       // every seq q_len==1 → Flash-Decoding
    Ragged,                                           // variable q_len>=1 → tile-scheduled kernel
    Spec { mask: MaskMode, mask_handle: Option<MaskHandle> }, // linear chain now (mask=Causal, handle=None)
}

/// One forward step's plan — PURE host-side metadata + schedule, backend-agnostic (inv 4).
/// DECISION (resolving the duplicate definitions): device-resident index tensors do NOT live here;
/// they live in `KvIndexTensors` (device, on the scope). `BatchPlan` holds host Vecs only. The
/// runtime uploads them into `KvIndexTensors` once per step.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    pub kind: BatchKind,
    pub num_tokens: usize,
    pub batch: usize,
    pub q_lens: Vec<i32>,          // per-seq q_len; decode = all 1 (dissolves "1-slot-per-row")
    pub kv_lens: Vec<i32>,         // per-seq KV length AFTER this step
    pub seq_positions: Vec<i32>,   // first cache row each seq writes (== kv_len BEFORE)
    pub rope_positions: Vec<i32>,  // absolute RoPE position per token
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub total_q_tiles: i32,        // ragged tile count (0 for DecodeOnly)
}
impl BatchPlan {
    pub fn is_decode_only(&self) -> bool;
    /// Host-side prefix-sum + ragged tile schedule (every backend uploads the same schedule).
    pub fn plan_ragged_tiles(q_lens: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>); // (cu_q_lens, block2req, block2tile)
}

/// Per-seq host description fed by callers (today's `SeqStep`, unchanged shape).
#[derive(Debug, Clone)]
pub struct SeqStep { pub sequence_id: SeqId, pub input_ids: Vec<i32>, pub positions: Vec<i32>, pub kv_write_start: i32, pub kv_len_after: i32, pub block_table: Vec<u32> }

/// One token result carrying probability info (structural fix for "argmax-only, no probabilities").
#[derive(Debug, Clone)]
pub struct SampledToken { pub token_id: i32, pub logprob: f32, pub top_logprobs: Vec<(i32, f32)> }

/// Uniform runtime → step seam (consumed by `Runtime::step`).
#[derive(Debug, Clone)]
pub struct StepRequest {
    pub seqs: Vec<SeqStep>,
    pub sampling: Vec<crate::domain::ports::sampler::SamplingParams>, // routed to the Sampler strategy
    pub stop: StopCriteria,
    pub draft_tokens: Vec<Vec<i32>>, // spec verify only; empty otherwise
}

/// Uniform step result — the SINGLE commit source (inv 9). `accepted[i]` REPLACES `kv_len += 1`:
/// decode=1, accepted prefill=q_len, spec verify=1..=K. ONE element type (`SampledToken`) carries
/// logprobs — no parallel `logprobs` vec.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub tokens: Vec<Vec<SampledToken>>, // per-seq, ragged by accepted[i]
    pub accepted: Vec<u32>,
    pub finished: Vec<bool>,
    pub hidden_tap: Option<HiddenTap>, // EAGLE/PP export; None on ordinary decode (reserved-but-wired)
}

#[derive(Debug, Clone)]
pub struct StopCriteria { pub eos_ids: Vec<i32>, pub generated_counts: Vec<u32>, pub max_tokens: Vec<u32>, pub ignore_eos: Vec<bool> }

/// Captured hidden state for EAGLE/PP, opaque to the runtime.
pub struct HiddenTap { pub at_layer: usize /* + the Hidden stream, dtype/shape erased for transport */ }
```

```rust
// ── kv.rs — PagedKvPool + KvView (one view type) + KvEdit (mutation). The pool owns per-seq kv_len
//    so KvEdit is self-contained (inv 9). KvIndexTensors fold INTO the view (per critique). ──
use crate::domain::exec::Device;
use crate::domain::dtype::quant::QuantScheme;

/// RESERVED KV-cache quant tier. `None` ships (full-precision, byte-identical). New scheme reuses the
/// shared `QuantScheme`; never a new pool method.
#[derive(Debug, Clone)]
pub enum KvQuantTier {
    None,
    PerTensor(QuantScheme),                                  // available-now tier (fp8/int8 KV)
    PerBlock { scheme: QuantScheme /*, per-block scale pool — reserved */ }, // RESERVED (§6)
}

/// One transformer layer's slice. Shape `[num_blocks, block_size, kv_dim]` for K and V.
pub struct PagedKvLayer<T: Dtype, D: Device> { pub k: Tensor<T, D>, pub v: Tensor<T, D> }

/// Device-resident per-step index tensors (today's block_tables/cu_q_lens/... de-coupled from the
/// plan so the plan stays backend-agnostic). Built per step by the runtime, lives on the scope.
pub struct KvIndexTensors<D: Device> {
    pub block_tables: Tensor<i32, D>,   // [batch, max_blocks_per_seq]
    pub cu_q_lens: Tensor<i32, D>,      // [batch+1]
    pub kv_lens: Tensor<i32, D>,        // [batch]
    pub seq_positions: Tensor<i32, D>,  // [batch]
    pub seq_lens_step: Tensor<i32, D>,  // [batch] (== q_len[i])
    pub rope_positions: Tensor<i32, D>, // [num_tokens]
    pub block2req: Tensor<i32, D>,      // ragged schedule (placeholders for DecodeOnly)
    pub block2tile: Tensor<i32, D>,
}

/// Worker-owned paged pool (inv 9 substrate). Owns per-seq `kv_len`. Borrowed per layer-range as a
/// `KvView`; mutated through `KvEdit`.
pub struct PagedKvPool<T: Dtype, D: Device> {
    pub layers: Vec<PagedKvLayer<T, D>>,
    pub num_blocks: usize, pub block_size: usize, pub kv_dim: usize,
    pub quant: KvQuantTier,                 // default None
    pub seq_kv_len: std::collections::HashMap<SeqId, u32>, // pool-owned per-seq length (inv 9)
}
impl<T: Dtype, D: Device> PagedKvPool<T, D> {
    pub fn num_layers(&self) -> usize;
    /// Borrow `range`'s layers + the step's device index tensors as a single `KvView`. PP runs
    /// `decode_layers(range)` so the view is range-scoped.
    pub fn view<'a>(&'a mut self, range: LayerRange, index: &'a KvIndexTensors<D>) -> KvView<'a, T, D>;
    /// Open a mutation transaction for the commit phase.
    pub fn edit<'a>(&'a mut self) -> KvEdit<'a, T, D>;
}

/// The SINGLE paged view a layer-range presents (read AND write). Holds `&mut [PagedKvLayer]`; a layer
/// derives a per-layer `LayerKv` (mutable) for `scatter_kv_paged` and reads the same storage for
/// `attention_paged` — scatter-then-attend within one layer is a borrow split on this one type (no
/// separate `KvViewMut`).
pub struct KvView<'a, T: Dtype, D: Device> {
    pub layers: &'a mut [PagedKvLayer<T, D>],
    pub index: &'a KvIndexTensors<D>,
    pub num_blocks: usize, pub block_size: usize, pub kv_dim: usize,
    pub quant: &'a KvQuantTier,
}
impl<'a, T: Dtype, D: Device> KvView<'a, T, D> {
    /// Mutable write handle for a single layer (fed to `scatter_kv_paged`).
    pub fn layer_mut(&mut self, layer_idx: usize) -> LayerKv<'_, T, D>;
    /// Read-only K/V for a single layer (fed to `attention_paged` after scatter completes).
    pub fn layer(&self, layer_idx: usize) -> (&Tensor<T, D>, &Tensor<T, D>);
}

/// One layer's mutable K/V + indices, for the scatter write path.
pub struct LayerKv<'a, T: Dtype, D: Device> { pub k: &'a mut Tensor<T, D>, pub v: &'a mut Tensor<T, D>, pub index: &'a KvIndexTensors<D> }

/// Per-seq KV mutation plan — the structural replacement for `seq.kv_len += 1` + one-block push
/// (decode_engine.rs:264). Built from `StepOutput.accepted`. The pool owns kv_len, so this is
/// self-contained. `rollback` is NOT a method (it is `truncate(kv_len - n)`); add it only if chained
/// spec ever needs the sugar (§6).
pub struct KvEdit<'a, T: Dtype, D: Device> { pub pool: &'a mut PagedKvPool<T, D> }
impl<'a, T: Dtype, D: Device> KvEdit<'a, T, D> {
    /// Commit `n` newly-written rows for `sid` (decode n=1; spec n=accepted). Allocates new physical
    /// blocks only when kv_len crosses a block boundary — multi-slot growth, not one-per-row.
    fn append(&mut self, sid: SeqId, n: u32) -> OpResult<()>;
    /// Truncate `sid` so its kept KV length becomes exactly `to`. Spec rejection: after writing the
    /// candidate run and accepting k, `truncate(sid, base + k)` discards the rejected tail. Returns
    /// freed physical blocks for the allocator.
    fn truncate(&mut self, sid: SeqId, to: u32) -> OpResult<Vec<u32>>;
    /// Apply a whole step's accepted counts in one pass (what `decode_engine` calls). Returns freed blocks.
    fn apply_step(&mut self, sids: &[SeqId], accepted: &[u32], speculative_len: &[u32]) -> OpResult<Vec<u32>>;
}

/// Stable per-sequence identity (matches the scheduler's `sequence_id: u64`).
pub type SeqId = u64;
```

### 3.6 Sampling — `domain/ports/sampler.rs`

```rust
use crate::domain::exec::StepCtx;
use crate::domain::tensor::Tensor;
use crate::domain::dtype::Dtype;
use crate::domain::ports::{backend::LlmBackend, error::OpResult};
use crate::domain::plan::SampledToken;

/// Per-request sampling configuration. Open: a new strategy = a new field with a neutral default,
/// never a new method. `temperature == 0.0` ⇒ greedy (degenerates to argmax, the current path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32, pub top_k: u32, pub top_p: f32, pub min_p: f32,
    pub repetition_penalty: f32, pub seed: Option<u64>, pub want_logprobs: bool,
}
impl Default for SamplingParams { fn default() -> Self; } // greedy-equivalent neutral defaults

/// One sampling call result: one `SampledToken` per sampled row (located via `ctx.plan().q_lens`).
#[derive(Debug, Clone, Default)]
pub struct SampleBatch { pub tokens: Vec<SampledToken> }

/// Speculative verify verdict. Linear chain now; tree/Medusa reserved via `BatchKind::Spec` (NOT here).
#[derive(Debug, Clone)]
pub struct AcceptReject {
    pub accepted_count: Vec<u32>,        // feeds KvEdit (inv 9)
    pub bonus_token: Vec<SampledToken>,  // residual (target − draft) sample at the rejection point
}

/// Pluggable, probability-returning token selection. A STRATEGY trait, NOT a backend capability — it
/// is implemented on marker types (`GreedySampler`, `ChainSampler`), so it is NOT in the `Backend`
/// alias (inv 10). OBJECT-SAFE by construction: the logits dtype is the model's concrete `T` (fixed
/// inside the monomorphized `run::<D,T,M>` entry), so `Box<dyn Sampler<T, D>>` is legal — there is NO
/// `DynSampler`/`ErasedLogits` (both removed). All bodies default-compose from `MathOps::softmax`, so
/// greedy is byte-identical to today's argmax and a new backend writes ZERO sampling code.
pub trait Sampler<T: Dtype, D: LlmBackend>: Send + Sync {
    /// Sample one token per sequence. The scope rides in `ctx` (enqueued on its stream AFTER the
    /// captured graph region, which is what lets temperature/top-p coexist with graph capture).
    /// default (greedy): argmax of per-seq last row + log_softmax logprob.
    fn sample(&self, logits: &Tensor<T, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<SampleBatch>;
    /// Per-row probability rows (post temperature/top-k/top-p) WITHOUT drawing (spec verify + logprobs).
    /// default: softmax(scaled_logits) after in-place param filters.
    fn probs(&self, logits: &Tensor<T, D>, params: &[SamplingParams], out: &mut Tensor<f32, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    /// Speculative accept/reject (linear chain). default: per-position ratio test
    /// `r = target_prob[tok]/draft_prob[tok]`, accept if `u < min(1,r)` else stop + residual sample.
    /// Mask-agnostic: tree mode reads `ctx.plan().kind` — this signature never reshapes.
    fn verify(&self, target_logits: &Tensor<T, D>, draft_tokens: &[i32], draft_probs: &Tensor<f32, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<AcceptReject>;
}
```

```rust
// ── application/sampler_stack.rs — concrete strategies (zero-cost markers) ──
/// Greedy/argmax: the default path (every row temperature==0). Inherits all trait defaults.
pub struct GreedySampler;
impl<T: Dtype, D: LlmBackend> Sampler<T, D> for GreedySampler {}

/// Temperature + top-k + top-p + min-p multinomial. CUDA may override `sample` with a fused kernel;
/// CPU inherits the softmax-composed default.
pub struct ChainSampler;
impl<T: Dtype, D: LlmBackend> Sampler<T, D> for ChainSampler {
    fn sample(&self, logits: &Tensor<T, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<SampleBatch>;
}
// The runtime holds `Box<dyn Sampler<T, D>>` (permitted dyn seam (b), inv 11), selected from params.
```

### 3.7 Dtype & quant — `domain/dtype/{mod,quant}.rs`, `domain/tensor.rs`

```rust
// ── dtype/mod.rs — OPEN dtype (inv 8). Replaces closed `DataType` + numel*SIZE_BYTES dyn paths. ──
use half::{bf16, f16};

/// Open dtype identity (registry key, not a closed match-arm set). Built-ins occupy reserved low ids;
/// a new scalar registers additively. Equality/hash by id. Invariant: a `DTypeId` is a STORAGE scalar
/// of fixed byte width — sub-byte logical types (int4/mxfp4) are NEVER `DTypeId`s (they are a
/// `QuantScheme.packing` over a byte dtype reached via `bitcast`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeId(pub u16);
impl DTypeId {
    pub const F32: DTypeId = DTypeId(0);  pub const F16: DTypeId = DTypeId(1);  pub const BF16: DTypeId = DTypeId(2);
    pub const I32: DTypeId = DTypeId(3);  pub const I8:  DTypeId = DTypeId(4);
    pub const F8E4M3: DTypeId = DTypeId(5); pub const F8E5M2: DTypeId = DTypeId(6);
    pub const U8:  DTypeId = DTypeId(7);  pub const U32: DTypeId = DTypeId(8); // bitcast carriers for packed payloads
    /// Register a novel scalar (heterogeneous backend); returns a fresh id above the reserved range.
    pub fn register(spec: DTypeSpec) -> DTypeId;
    /// Byte width — single source of truth for size in dyn-typed paths (loaders / binary edge).
    pub fn size_bytes(self) -> usize;
    pub fn is_float(self) -> bool;
}
#[derive(Clone, Copy, Debug)] pub struct DTypeSpec { pub size_bytes: usize, pub is_float: bool, pub name: &'static str }

/// Compile-time dtype trait (spine kept — inv 8): `SIZE_BYTES` + read/write_f64 unchanged, so
/// `Tensor<T,D>` monomorphization and `numel*SIZE_BYTES` math are untouched. `DATA_TYPE` is replaced
/// by the open `ID`.
pub trait Dtype: Copy + Send + Sync + 'static + std::fmt::Debug {
    const ID: DTypeId;
    const SIZE_BYTES: usize;
    fn read_f64(raw: &Self) -> f64;   // lossy widen for host dequant/debug
    fn write_f64(v: f64) -> Self;     // lossy narrow for host quant/init
}
// Built-ins keep identical SIZE_BYTES:
impl Dtype for f32  { const ID: DTypeId = DTypeId::F32;  const SIZE_BYTES: usize = 4; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for f16  { const ID: DTypeId = DTypeId::F16;  const SIZE_BYTES: usize = 2; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for bf16 { const ID: DTypeId = DTypeId::BF16; const SIZE_BYTES: usize = 2; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for i32  { const ID: DTypeId = DTypeId::I32;  const SIZE_BYTES: usize = 4; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for i8   { const ID: DTypeId = DTypeId::I8;   const SIZE_BYTES: usize = 1; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for u8   { const ID: DTypeId = DTypeId::U8;   const SIZE_BYTES: usize = 1; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for u32  { const ID: DTypeId = DTypeId::U32;  const SIZE_BYTES: usize = 4; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }

/// fp8 storage scalars (1 byte), first-class NOW. Opaque byte newtypes (GPU reality: compute via
/// upcast); host read/write_f64 cover quant/dequant/debug.
#[derive(Clone, Copy, Debug)] #[repr(transparent)] pub struct Fp8E4m3(pub u8);
#[derive(Clone, Copy, Debug)] #[repr(transparent)] pub struct Fp8E5m2(pub u8);
impl Dtype for Fp8E4m3 { const ID: DTypeId = DTypeId::F8E4M3; const SIZE_BYTES: usize = 1; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }
impl Dtype for Fp8E5m2 { const ID: DTypeId = DTypeId::F8E5M2; const SIZE_BYTES: usize = 1; /* ... */ fn read_f64(r:&Self)->f64; fn write_f64(v:f64)->Self; }

pub trait Float: Dtype {}
impl Float for f32 {} impl Float for f16 {} impl Float for bf16 {} impl Float for Fp8E4m3 {} impl Float for Fp8E5m2 {}
```

```rust
// ── dtype/quant.rs — QuantScheme descriptor (inv 8). Replaces bare group_size. ──
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Granularity { PerTensor, PerChannel, PerGroup }
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub enum Symmetry    { Symmetric, Asymmetric }
/// Physical bit-packing of the weight payload — the ONLY place a kernel learns "how to unpack".
/// New family (fp4/int3/GPTQ variant) = one new arm; no signature anywhere changes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Packing {
    AwqInt4,   // 8×int4 LE-packed into a u32 word; AWQ scale/zero layout
    GptqInt4,  // RESERVED (column-interleaved + g_idx)
    Int8,      // one scalar/elem, no sub-byte packing
    Fp8,       // fp8 weights + separate scale tensor; no unpack
    Mxfp4,     // RESERVED OCP MXFP4 (32-wide fp4 blocks + e8m0 block scale)
}
/// ONE value carrying everything a quant kernel needs beyond the tensors. Replaces `group_size`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuantScheme { pub granularity: Granularity, pub symmetry: Symmetry, pub packing: Packing, pub group: usize }
impl QuantScheme {
    /// Canonical AWQ int4 (the only scheme wired today): PerGroup{128}·Asymmetric·AwqInt4.
    pub const AWQ_INT4_G128: QuantScheme = QuantScheme { granularity: Granularity::PerGroup, symmetry: Symmetry::Asymmetric, packing: Packing::AwqInt4, group: 128 };
    /// Logical K elements per packed storage scalar (8 for int4-in-u32, 1 for int8/fp8) — replaces the
    /// hardcoded `wp_shape[1] * 8` in CUDA matmul_quant.
    pub fn logical_per_word(self) -> usize;
}
```

```rust
// ── tensor.rs — the single tensor type + bitcast (inv 2, 8) ──
/// Spine kept; the device bound is `Device` (folds in memory). `bitcast` (declared on MathOps) gives
/// the zero-copy cross-dtype view that replaces int4-as-i32. (Signature in §3.2.)
pub struct Tensor<T: Dtype, D: Device> { /* Arc<Storage> + shape + dtype PhantomData<T> */ }
```

```rust
// ── components/quant_linear.rs — QuantLinear (was dead code), carries a QuantScheme value ──
pub struct QuantLinear<A: Dtype, W: Dtype, O: Dtype, D: MathOps> {
    pub weight: Tensor<W, D>,            // packed in TRUE storage dtype (u32 for AWQ int4, i8, fp8…)
    pub scales: Tensor<A, D>,
    pub zeros: Option<Tensor<W, D>>,     // present iff scheme.symmetry == Asymmetric
    pub scheme: QuantScheme,             // replaces the bare group_size
    pub _out: core::marker::PhantomData<O>,
}
impl<A: Dtype, W: Dtype, O: Dtype, D: MathOps> QuantLinear<A, W, O, D> {
    /// output = dequant(input × weight; scheme). // body: one D::matmul_quant call, no per-scheme branch
    pub fn forward(&self, scope: &D::Scope, input: &Tensor<A, D>, output: &mut Tensor<O, D>) -> OpResult<()>;
}
```

### 3.8 Runtime & host — `application/{runtime,hosting,spec_runtime,decode_engine}.rs`

```rust
// ── runtime.rs ──
use crate::domain::exec::{Device, ExecScope};
use crate::domain::ports::{backend::LlmBackend, sampler::Sampler, error::OpResult};
use crate::domain::model::{DecoderModel, ModelDims, SampleRows};
use crate::domain::component::{Hidden, LayerRange};
use crate::domain::kv::{PagedKvPool, KvIndexTensors};
use crate::domain::plan::{BatchPlan, StepRequest, StepOutput};
use crate::domain::dtype::Dtype;

/// Drives a single `DecoderModel` over one device + one `ExecScope`. Owns the paged KV pool, the
/// address-stable `Hidden` slot (sized once for CUDA-graph stability — inv 7), the device index
/// tensors, the sampler strategy, and (CUDA only) the graph runner. Generics mirror today's
/// `ModelRunner<T,D,M>`; `M: DecoderModel`; the workspace god-struct is gone.
pub struct Runtime<T, D, M>
where T: Dtype, D: LlmBackend, M: DecoderModel<T, D> {
    pub model: M,
    pub kv_pool: PagedKvPool<T, D>,
    pub kv_index: KvIndexTensors<D>,    // reusable device index buffers (no per-step malloc)
    pub hidden: Hidden<T, D>,           // address-stable residual slot, allocated once (inv 7)
    pub scope: <D as Device>::Scope,    // the unified concurrency+device-activation seam (inv 5)
    pub sampler: Box<dyn Sampler<T, D>>,// permitted dyn seam (b) (inv 11)
    pub dims: ModelDims,
    pub block_size: usize, pub max_blocks_per_seq: usize, pub max_seq_len: usize,
    pub cap_num_tokens: usize, pub cap_batch: usize, pub capture_sizes: Vec<usize>,
    pub graph: Option<GraphRunner<D>>,  // None until primed → eager (today's semantics, generalized)
}

impl<T, D, M> Runtime<T, D, M>
where T: Dtype, D: LlmBackend, M: DecoderModel<T, D> {
    pub fn new(model: M, scope: <D as Device>::Scope, sampler: Box<dyn Sampler<T, D>>, num_blocks: usize, block_size: usize, max_blocks_per_seq: usize, max_seq_len: usize, cap_num_tokens: usize, cap_batch: usize, capture_sizes: Vec<usize>) -> OpResult<Self>;

    /// THE seam. Resolve `req` → `BatchPlan` (+ upload `kv_index`), pick a `GraphDecision`, run
    /// embed → decode_layers(all) → finalize under the scope, sample, return a `StepOutput` whose
    /// `accepted[]` drives the KV commit. No `kv_len += 1` here.
    ///
    /// BORROW NOTE (inv 9 composition): destructure disjoint fields —
    /// `let Runtime { kv_pool, kv_index, hidden, scope, model, sampler, .. } = self;` — so `&scope`
    /// (for StepCtx) and `&mut kv_pool` (for KvView) are independent borrows. `StepCtx` MUST NOT
    /// borrow `kv_pool`.
    pub fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput>;

    /// Eager path: always available (CPU has only this). Fallback for MoE/spec/Ragged and
    /// un-captured shapes. Mirrors today's `step_batch_eager`.
    pub fn step_eager(&mut self, plan: &BatchPlan, req: &StepRequest) -> OpResult<StepOutput>;

    /// Fast/slow router. DecodeOnly (incl. dropless fixed-capacity MoE, §6) may replay a captured
    /// graph; Ragged/Spec/over-capacity → Eager. Enum (not bool) so future fast paths add variants.
    pub fn decide(&self, plan: &BatchPlan) -> GraphDecision;

    /// CUDA-only graph warm+capture for `capture_sizes`. No-op when the Scope reports no graph support.
    pub fn prime_graphs(&mut self) -> OpResult<()>;
}

#[derive(Debug, Clone, Copy)] pub enum GraphDecision { Graph(GraphSlotId), Eager }
#[derive(Debug, Clone, Copy)] pub struct GraphSlotId(pub usize);
/// Backend-generic graph runner; non-CUDA scopes report `supports_graphs()==false` → stays None.
pub struct GraphRunner<D: LlmBackend> { _d: core::marker::PhantomData<D> }
```

```rust
// ── hosting.rs — co-host 1..N models in one ExecScope (spec draft+target; PP stages) ──
/// Architecture-erased runtime facade. Erases ONLY `M` (the arch), never `D`/`T`, so the op spine
/// stays monomorphized while the host holds heterogeneous architectures (target=Llama3, draft=Qwen3).
/// This is the permitted dyn seam (a) (inv 11).
pub trait ErasedRuntime<T: Dtype, D: LlmBackend>: Send {
    fn step(&mut self, req: &StepRequest) -> OpResult<StepOutput>;
    /// Run a layer sub-range and tap `Hidden` (PP send / EAGLE feed).
    fn run_layers(&mut self, range: LayerRange, req: &StepRequest) -> OpResult<crate::domain::plan::HiddenTap>;
    fn prime_graphs(&mut self) -> OpResult<()>;
    fn dims(&self) -> &ModelDims;
}
// Blanket: impl<T,D,M> ErasedRuntime<T,D> for Runtime<T,D,M> { ... }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum ModelRole { Draft, Target, EagleHead, PipelinePeer }

/// Container for co-hosted models sharing a device + scope. `primary` is concrete on the hot path
/// (the macro entry gives a concrete `Runtime<T,D,M>`); `aux` uses the erased facade for differing
/// architectures (e.g. a draft model).
pub struct ModelHost<T: Dtype, D: LlmBackend> {
    pub primary: Box<dyn ErasedRuntime<T, D>>,                 // dyn seam (a)
    pub aux: Vec<(ModelRole, Box<dyn ErasedRuntime<T, D>>)>,
    pub topology: crate::domain::exec::TopologyShape,         // single-rank default
}
```

```rust
// ── spec_runtime.rs (linear chain now; tree reserved) ──
pub struct SpecRuntime { pub num_draft: usize }
impl SpecRuntime {
    /// Draft proposes `num_draft` tokens/seq; target verifies in ONE Ragged/Spec StepRequest;
    /// `Sampler::verify` yields per-seq accepted → `KvEdit` truncates the rejected suffix. Tree mode
    /// flips `BatchKind::Spec.mask` + adds a kernel arm — this signature never changes.
    pub fn step_request<T: Dtype, D: LlmBackend>(&mut self, host: &mut ModelHost<T, D>, req: &StepRequest) -> OpResult<StepOutput>;
    /// Compatibility wrapper for scheduler-owned active state; production callers should build
    /// `StepRequest` and use `step_request` so verification has tokens/positions/block tables.
    pub fn step<T: Dtype, D: LlmBackend>(&mut self, host: &mut ModelHost<T, D>, active: &mut ActiveSeqMap) -> OpResult<StepOutput>;
}

// ── decode_engine.rs (commit by accepted count, not +1) ──
pub struct DecodeEngine { /* rows */ }
impl DecodeEngine {
    /// One decode/spec step + accepted-count commit. Replaces the `seq.kv_len += 1; block_table.push`
    /// loop (decode_engine.rs:264-266) with `pool.edit().apply_step(&sids, &out.accepted, &spec_len)`.
    pub fn run_step<T: Dtype, D: LlmBackend>(&mut self, host: &mut ModelHost<T, D>, spec: Option<&mut SpecRuntime>, active: &mut ActiveSeqMap, /* prefilling, kv_allocator, control, data, stop */) -> OpResult<()>;
}
```

---

## 4. Binary-edge dispatch

**Macro-generated `match` over the SHIPPED `(backend, dtype, arch)` tuple set — exactly one
fully-typed entry per realized combo; the only `dyn`s are the two off-the-op-path seams from inv 11.**

`worker_main` resolves `(backend_id, dtype_id, arch_id)` at runtime (backend from build/CLI;
dtype + arch from `config.json` via `resolve_model_type`, as today). A macro generates a single
`match` whose arms each call one typed entry `run::<D, T, M>(...)`. **Only combos we actually ship
get monomorphized** (e.g. `Cuda × bf16 × {Llama3, Qwen3}`); all other points are compile-time absent.
This formalizes today's ad-hoc `match model_type` (worker_main.rs:242, which hardcodes `<bf16, Cuda>`)
into an explicit, additively-extended table: shipping a new arch/dtype/backend = adding ONE macro row.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum BackendId { Cuda, Cpu /*, Rocm, Metal, Vulkan */ }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum DTypeSel  { Bf16, F16, Fp8E4m3 /*, Int4Awq via QuantScheme */ }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum ArchId    { Llama3, Qwen3 /*, Mixtral, DeepSeek */ }

/// The single fully-typed entry. Inside it `D`, `T`, `M` are concrete → the entire op path is
/// monomorphized and dyn-free. Loads weights, builds the `Runtime`/`ModelHost` (selecting the
/// `Box<dyn Sampler<T,D>>` strategy from config), sends Ready, runs the serve loop.
pub fn run<D, T, M>(boot: Bootstrap<'_>, control: &ControlPump, data: &DataPump) -> Result<(), String>
where D: LlmBackend, T: Dtype, M: DecoderModel<T, D>;

/// Expands the SHIPPED tuple table into one `match`. Unshipped points are compile-time absent.
#[macro_export]
macro_rules! dispatch_worker {
    ( $sel:expr, $boot:expr, $control:expr, $data:expr;
      $( ($be:ident, $dt:ident, $arch:ident) => $d:ty, $t:ty, $m:ty );* $(;)? ) => {
        // match $sel {
        //   $( (BackendId::$be, DTypeSel::$dt, ArchId::$arch)
        //        => $crate::application::run::<$d,$t,$m>($boot,$control,$data), )*
        //   other => Err(format!("unshipped (backend,dtype,arch): {:?}", other)),
        // }
    };
}
// Example invocation (ONE row per shipped combo):
//   dispatch_worker!(sel, boot, control, data;
//     (Cuda, Bf16, Llama3) => Cuda, bf16, Llama3Model<bf16, Cuda>;
//     (Cuda, Bf16, Qwen3)  => Cuda, bf16, Qwen3Model<bf16, Cuda>;
//   );
```

Why no blowup: the macro never enumerates the full product space — only authored rows exist. Why no
op-path `dyn`: `run::<D,T,M>` fixes `D` and `T`, so `Box<dyn ErasedRuntime>` (arch-erased draft) and
`Box<dyn Sampler<T,D>>` (strategy) both keep `D`/`T` concrete; the op spine never sees a trait object.

---

## 5. Extension walkthroughs

Each lists the EXACT additive changes and confirms no existing signature breaks.

### 5.1 Speculative decoding
- Add a `ModelRole::Draft` entry to `ModelHost.aux` and instantiate `SpecRuntime { num_draft }`.
- `spec_runtime` runs the draft `decode_layers` to produce candidates; the target verifies them in one
  `Ragged`/`Spec` `StepRequest`; call `Sampler::verify` → `AcceptReject`.
- Commit reads `StepOutput.accepted[seq]`; `decode_engine` calls `KvEdit::apply_step` (which `truncate`s
  the rejected suffix) instead of `seq.kv_len += 1`.
- **No signature breaks:** `accepted`/`hidden_tap` already on `StepOutput`; `verify` is mask-agnostic;
  tree/Medusa is the reserved `BatchKind::Spec { mask: MaskMode::Tree, mask_handle: Some(_) }` (§6).

### 5.2 MoE
- Set `ModelDims.num_experts/experts_per_tok/moe_intermediate_size > 0` in the loader.
- Build the block as `DecoderBlock<_, _, MoeFfn>` instead of `DenseFfn` (one type at model-build time).
- Add `Router` top-k usage + `FusedOps::grouped_expert_gemm` (default already MathOps-composed); CUDA
  adds a `#[must_override]` grouped/permuted kernel.
- Expert intermediates live in `MoeFfn`'s private scratch (no shared role enum, no workspace reshape).
- **No signature breaks:** dense models (`num_experts == 0`) are byte-identical; `Component`/`DecoderModel`
  signatures unchanged. Graph-replay for MoE decode is available IFF experts are dropless/fixed-capacity
  (§6). Expert-parallel later = add `CollectiveOps::all_to_all(Ep, ...)` inside `MoeFfn::run`.

### 5.3 Tensor parallelism
- At load, tag each weight with a `ShardSpec` (q/k/v/gate/up = `ColumnParallel`; o_proj/down_proj =
  `RowParallel`) and load via `ShardedLoad::load_shard`.
- Insert ONE `CollectiveOps::all_reduce(scope, Tp, Sum, &mut out)` at the fixed row-parallel seam inside
  `Attention`/`Ffn` `Component::run`.
- The seam reads its position from `scope.rank().tp_rank` (a field, inv 11).
- **No signature breaks:** single-rank `all_reduce` is the identity no-op; `tp_size == 1` `load_shard`
  returns full tensors. No op/component/model signature changes.

### 5.4 Pipeline parallelism
- Compute the band via `LayerRange::for_pp_rank(pp_rank, pp_size, num_layers)` and pass it to the existing
  `decode_layers(range, ...)`.
- Guard `embed` on `topology.is_pp_first()`, `finalize` on `is_pp_last()`.
- At band edges, `CollectiveOps::recv(Pp, prev)` then `send(Pp, next)` of `Hidden.stream` (overlap later
  via the reserved `ExecScope::fork`/`record_event`/`wait_event`, §6).
- **No signature breaks:** `Hidden` is already a carried value; `LayerRange` already a parameter;
  `ErasedRuntime::run_layers` already exposes the tap. `StepCtx` unchanged.

### 5.5 Multi-node
- Add a new `CollectiveOps` impl (NCCL+IB / gloo) behind a feature flag and fill the reserved
  communicator-bootstrap body (§6).
- Populate `Rank.node_rank/node_size` + `TopologyShape.node` from a larger `WorkerGroup`.
- **No signature breaks:** call sites unchanged — purely an additive impl + one bootstrap body; the scope
  reads `node_rank` from its `Rank` field.

### 5.6 Heterogeneous backend (ROCm / Metal / Vulkan / CPU-SIMD)
- Add `infrastructure/<backend>/`: `impl Device` (choose `type Scope`, `type Stream`), the memory surface,
  and a `<Backend>Scope: ExecScope` whose `enter()` activates the native context.
- `impl MathOps` (~15 fns). `FusedOps`/`DiffusionOps` are inherited via default-compose; un-kerneled arms
  return `OpError::unsupported("<backend>", "...")`.
- `CollectiveOps` = the single-rank no-op identity.
- Add `BackendId::<Backend>` + one macro row.
- **No signature breaks:** zero edits to any existing trait or backend; `must_override` keeps CUDA from
  silently inheriting a slow default while the new backend inherits freely.

### 5.7 Quantization (fp8 / int4 / mxfp4 + KV-quant)
- **fp8 (now):** `Fp8E4m3`/`Fp8E5m2` are already `Dtype`/`Float`; pass `QuantScheme { packing: Fp8, .. }`;
  a backend lights it up with an fp8 branch inside its existing `matmul_quant`/`cast`.
- **int4 AWQ:** load packed weight as `Tensor<u32, D>`, use `QuantScheme::AWQ_INT4_G128`; `bitcast`
  exposes the byte/word view a kernel wants (replaces int4-as-i32). `QuantLinear` carries the scheme.
- **GPTQ / mxfp4:** add one `Packing` arm (`GptqInt4`/`Mxfp4`, already reserved) + its unpack branch
  inside `matmul_quant`/`grouped_expert_gemm`.
- **KV-quant:** construct `PagedKvPool { quant: KvQuantTier::PerTensor(scheme), .. }`; attention/scatter
  read `kv.quant` and branch. Per-block is the reserved `KvQuantTier::PerBlock` (§6).
- **No signature breaks:** every quant feature is a new `Packing`/`Dtype`/tier arm — never a new method.
  `matmul_quant`/`grouped_expert_gemm` already take `&QuantScheme`; the KV path already carries the tier.

---

## 6. Reserved seams

ABIs shaped now, deliberately unimplemented. Each: the signature already accommodates it; only a body /
kernel branch is deferred.

- **Tree/Medusa mask** — `BatchKind::Spec { mask: MaskMode::Tree, mask_handle: Option<MaskHandle> }` is the
  single reserved home (NOT also a raw pointer on `StepCtx`). `attention_paged` already routes on
  `plan.kind`; enabling trees adds a `MaskMode::Tree` kernel branch + a non-`None` `MaskHandle` producer.
  `Sampler::verify` is mask-agnostic. Deferred: the tree-attention kernel + mask registry.
- **NCCL / gloo collective bodies** — `CollectiveOps`'s `Comm` associated type and all method signatures are
  fixed; single-rank ships as identity. Deferred: one communicator-bootstrap fn body (the only place NCCL
  init lives) + the per-method NCCL calls. No call-site changes when it lands.
- **Per-block KV quant** — `KvQuantTier::PerBlock { scheme }` is reserved on `PagedKvPool.quant`; `KvView`
  already borrows `&KvQuantTier`. Deferred: the per-block scale pool + the scatter/gather quant branch. No
  `KvEdit`/`KvView`/attention signature change.
- **Expert-parallel all-to-all** — `CommAxis::Ep` + `CollectiveOps::all_to_all` are reserved; `MoeFfn::run`
  is the call site. Deferred: the dispatch/combine permute. The dropless single-GPU default stands until then.
- **CUDA-graph for variable shapes** — `GraphDecision` is an enum (not a bool) so future fast paths
  (captured-Ragged, fused-spec, MoE-with-fixed-capacity) attach as new variants. Today only
  `Graph(slot)`/`Eager` are emitted; **MoE decode is graph-capturable only if experts run dropless with
  fixed per-expert capacity (shape-stable)** — this is a stated requirement of `MoeFfn`, not silent
  additivity. Deferred: the variable-shape capture variants.
- **Multi-stream / cross-stage overlap** — `ExecScope::fork` / `record_event` / `wait_event` / `Event` are
  named as a reserved seam (commented out on the trait so v1 carries no dead methods). PP overlap and
  copy/compute overlap add them when implemented. Single-stream ships today.
- **Reduce ops beyond Sum / `ReduceOp::{Max,Min,Avg}`, `broadcast`, `barrier`, `ShardSpec::VocabParallel`,
  `KvEdit::rollback` sugar** — reserved on the surface so future sampling-side gathers, vocab-parallel
  lm_head, and chained-spec convenience are match-arms / one method, not reshapes.

---

## 7. Migration order

Each step compiles. Mapped to the 8 blockers. **(A)** = pure-additive; **(C)** = touches existing call sites.

1. **(A) Open dtype** — introduce `DTypeId` registry + keep `Dtype` spine; add `Fp8E4m3`/`Fp8E5m2`. Map the
   closed `DataType` (domain/types.rs) onto reserved ids; leave existing `impl Dtype` intact. *(Blocker 8)*
2. **(C) Fold `MemoryPort` into `Device`; rebound `Tensor<T, D: Device>`** — change `domain/tensor.rs:20`
   (`D: MemoryPort` → `D: Device`) and the device impls. Mechanical; touches every tensor-bound signature
   but no logic. *(Blocker 5 foundation, inv 2)*
3. **(C) `ExecScope` seam** — introduce `<D as Device>::Scope`, move `CudaConfig` internals behind
   `CudaScope`, replace `tensor.device().config.stream` and the diffusion thread-local with explicit
   `scope` args; `enter()` performs `cudaSetDevice`. *(Blockers 5, 7)*
4. **(C) De-parameterize `BatchPlan`** — `domain/batch.rs:34` (`BatchPlan<D: MemoryPort>` → `BatchPlan`);
   move device index tensors into `KvIndexTensors`; introduce `StepCtx`. Update fused-op call sites to take
   `ctx`. *(Blocker 6)*
5. **(C) Split op ports** — `MathOps` (floor) + `FusedOps`/`DiffusionOps` (default-composed ceilings); fold
   diffusion `sdpa`/`silu` into `MathOps`; add `#[must_override]` markers. Re-express `attention_paged`/
   `scatter_kv_paged` against `KvView`/`LayerKv`. *(Blocker 6)*
6. **(A) `QuantScheme`** — replace the bare `group_size` in `matmul_quant`; add `bitcast`; revive
   `QuantLinear` carrying a scheme; store packed int4 as `u32`. *(Blocker 8)*
7. **(C) `Hidden` + `Component`/`DecoderModel`** — dissolve `forward_workspace.rs` god-struct into the
   carried `Hidden` slot (allocated once by `Runtime`); un-fuse llama3.rs:213-223 cross-layer norm into
   per-block norms + `fused_add_rmsnorm`; introduce `embed`/`decode_layers(range)`/`finalize`, with
   `forward` as the provided default. *(Blocker 4)*
8. **(A) `KvEdit` commit** — pool owns per-seq `kv_len`; replace `seq.kv_len += 1` (decode_engine.rs:264)
   with `KvEdit::apply_step` driven by `StepOutput.accepted`. *(Blocker 1)*
9. **(A) `Sampler` strategy** — lift argmax out of the captured graph into `Box<dyn Sampler<T, D>>`
   (Greedy default = byte-identical); add `probs`/`verify`. *(Blocker 2)*
10. **(A) `CollectiveOps` sidecar** — add the trait + single-rank identity impl + `TopologyShape`/`Rank` on
    the scope from the reserved `LoadModel` fields. No behavior change at rank 1. *(Blocker 3)*
11. **(C) Binary-edge macro** — replace `match model_type` (worker_main.rs:242) with `dispatch_worker!` over
    shipped `(backend, dtype, arch)` tuples; the typed `run::<D,T,M>` entry builds `Runtime`/`ModelHost`.
    *(elegance pillar)*

Steps 1, 6, 8, 9, 10 are pure-additive once their predecessors land; 2–5, 7, 11 touch call sites but carry
no feature logic.

---

## 8. Comparison note

This spine maps cleanly onto the field while choosing the variant that fits a Rust monomorphized engine.
The **floor/ceiling op split with default-composed fused ops** is `ggml-backend`'s "implement a small op
set, fall back to generic" idea, but static instead of vtable-dispatched; the **`Component`/stage-list +
carried `Hidden`** mirrors `candle`'s module composition and HF-style decoder blocks; **paged KV +
`BatchPlan`/ragged tiles + accept-count commit** is vLLM/SGLang continuous-batching and
speculative-decode machinery, with vLLM's `BlockManager` re-expressed as `KvEdit` mutations; the
**`CollectiveOps`/`TopologyShape` seam + `ShardSpec`** is TensorRT-LLM / Megatron TP/PP sharding reduced to
additive impls; and the **binary-edge tuple macro** is our answer to TensorRT-LLM's build-time engine
specialization without its ahead-of-time compile step. Static-dispatch-with-default-composed-fused-ops is
the right spine here because the engine already commits to `Tensor<T, D>` monomorphization for kernel
perf: a `dyn`-backend (ggml/llama.cpp style) would erase exactly the type information CUDA codegen needs,
while a fully hand-written per-backend stack (TensorRT-LLM style) would forfeit the "new backend = ~15
portable ops" additivity. Keeping the single op spine static and isolating the two unavoidable `dyn`s
(arch object, sampler strategy) at the binary edge gives both vLLM's feature reach and candle's
compositional clarity without either's dispatch tax on the hot path.
