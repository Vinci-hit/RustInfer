# RustInfer Worker — 架构 v2（框架重设计）

> **状态：约束性设计文档。** 这份文档约束 `crates/infer-worker` 的下一阶段。
> 它只包含**设计**：trait/type/enum 签名 + 文档注释 + 组合方式。
> **不包含功能实现**，不包含方法体（默认实现逻辑只以简短的 `// ...` 注释出现）。
> 各子系统实现都要以这里固定下来的名称和关系为准。

---

## 1. 设计目标与不变量

### 组织原则

**模型是一个可切分的阶段列表，通过单一线程执行作用域在狭窄后端之上驱动。**

worker 只有**三个**承重的变化轴，未来的所有特性都必须挂到其中某一个轴上：

- **阶段之间** —— 切分点是 `embed` → `decode_layers(range)` → `finalize`，其中 `Hidden`
  作为一等的*传递值*存在。→ 流水线并行、EAGLE / hidden-state 探针、speculative 的
  draft+target 共驻。
- **阶段内部** —— `Component`：`Attention`、`DenseFfn | MoeFfn`、`Norm`、`Embed`、`LmHead`。
  → MoE（组件替换）、张量并行（在 `Attention`/`Ffn` 内固定接缝处做 collective）。
- **阶段之下** —— 后端操作底座 + dtype/quant 描述符 + `ExecScope`。
  → 异构后端、fp8/int4/mxfp4 量化、多流 / 多设备。

每个被要求支持的轴，都是*在哪里切*、*实例化哪个组件*、或*选择哪个底座实现*的几何结果，
而不是对承重签名做重塑。

### 硬性不变量

这些规则保证扩展是增量式的。它们彼此联动检查，并作为整体成立。

1. **静态分发是性能主干。** `Tensor<T: Dtype, D: Device>` 和所有 op 方法都是
   以 `Self` 为类型的**关联函数**（没有 `&self`），按 `(backend, dtype)` 单态化。
   只允许**两个** `dyn` 接缝，二者都不在 op 路径上，也都位于二进制边界
   （见不变量 11）。op 数学路径上永远不能出现 `dyn`。

2. **只有一个基础张量约束。** `Device` *就是* 持有内存的 trait：它包含 `MemoryPort`
   的能力面。`Tensor<T, D: Device>` 是唯一的张量类型，任何持有/产出 `Tensor` 的 trait
   或 struct 都必须以 `Device`（或其子 trait）为约束。调用点不再区分 `Device` 和
   `MemoryPort` —— `MemoryPort` 已经并入 `Device`。

3. **必需底座要窄。** 新后端只实现 `MathOps`（约 15 个纯 tensor→tensor 操作）。
   每个 `FusedOps` / `DiffusionOps` 方法都有由 `MathOps` 组合出来的默认体，所以正确性
   是免费的，后端只为速度重写。

4. **可移植签名里不要有编排物理。** `MathOps` 方法只接收 tensor + 描述符 + 一个 `&Scope`
   —— 不带 `BatchPlan`，不带分页 KV 地址，不在签名里暴露设备 scratch。
   **`BatchPlan` 不是后端泛型**（始终是 `BatchPlan`，不是 `BatchPlan<D>`）。

5. **只有一个并发 + 设备激活接缝——显式 `&Scope`，不要 thread-local。** 所有 stream/queue
   访问和 `cudaSetDevice` 都走 `ExecScope`。标准 scope 类型是 `<D as Device>::Scope`。
   `ExecScope` 是对这个关联类型的**约束**，绝不写成 `ExecScope<D>`。进入时使用 RAII
   （`enter()` 会在其生命周期内激活设备）。scope 通过 `scope: &<D as Device>::Scope`
   **显式**传给 `MathOps`，并在 `&StepCtx` 内传给 fused ops / samplers。**没有
   `active_scope()` thread-local，也没有 `ScopeRef`**（二者都因不安全/仪式化被移除）。
   当前两个访问路径（`tensor.device().config.stream` 和 diffusion 的 thread-local）都废弃，
   统一改走这一条显式路径。

6. **接收者统一。** *所有* capability-trait 方法（`MathOps`、`FusedOps`、`DiffusionOps`、
   `CollectiveOps`）都用没有 `self` 接收者的关联函数，因此只要作用域里有类型参数 `D`，
   就能写 `D::op(...)`。

7. **`Hidden` 是传递值，不是 workspace 守护神结构体。** 模型永远不会拥有固定的 dense-only
   视图集合。跨层 norm fusion 在**抽象层面保持不融合**（每层拥有自己的 norm）；
   速度收益要通过可覆盖的 `FusedOps::fused_add_rmsnorm`（以及其他 fused 原语）显式重新拿回，
   **绝不能**作为隐藏的跨层契约。每个组件的 scratch 都私有，且从单一 scope `Workspace`
   中再分配。**没有中心化的 `ScratchRole` enum**（那会重新制造 god-struct 耦合）。

8. **开放 dtype + 描述符驱动量化。** `DataType` 变为开放的 `DTypeId` 注册表；`Dtype`
   保留 `SIZE_BYTES` + 读写能力。`QuantScheme {granularity, symmetry, packing, group}` 取代
   了裸露的 `group_size`。fp8 e4m3/e5m2 现在是第一类 `Dtype` 实现。新的量化 = 在一个
   `matmul_quant` 里消费的一个新的 `Packing` 分支，而不是新增方法。`bitcast` 视图取代
   int4-as-i32。

9. **做变更，不做自增。** KV 提交通过 `KvEdit` 完成（`append(n)` / `truncate(to)` +
   每个序列的 `accepted_count`）。硬编码的 `seq.kv_len += 1`（decode_engine.rs:264）以及
   每行一个槽位的分配都废除。pool 持有每个序列的 `kv_len`，因此 `KvEdit` 是自包含的。

10. **正交侧车。** `CollectiveOps`（单 rank 时恒等）是后端能力；
    `Sampler`（默认 greedy）是一个**策略**对象，**不是**后端能力，因此不放进
    `Backend` 别名里。单 rank / greedy 路径保持与今天字节级一致。

11. **GAT 基石 + 有界 dyn 数量。** `ExecScope` 以**字段**形式携带 rank/stream/quant-tier，
    绝不做成类型参数；`TopologyShape`（可 Copy 的 rank/size 数据）就是其中一种字段。
    communicator 句柄放在后端上（设备自己就是 `CollectiveOps`），通过关联函数访问，
    不是 scope 上的泛型字段。随着 TP/PP/multi-node/KV-tier 到来，scope 类型也绝不会
    增加第二个泛型。只允许的两个 `dyn` 接缝是：**(a)** 架构擦除的模型对象
    （draft/aux 模型用 `Box<dyn DecoderModel<T, D>>`），以及 **(b)** 运行时 sampler
    策略（`Box<dyn Sampler<T, D>>`）。二者都不在 op 路径上；二者都位于已单态化的
    `(D, T)` 入口内部。

12. **LLM 路径不依赖 diffusion 上限。** `DecoderModel`/`Runtime` 约束在 `LlmBackend`
    （`MathOps + FusedOps + CollectiveOps`）上，而**不是**约束在一个会把 `DiffusionOps`
    也拖进来的 god-alias 上。

> **必须覆盖纪律（不变量 3 的推论）：** 性能关键的 `FusedOps` 方法带有
> `#[must_override(tier = "perf")]` 标记。对于构建中属于性能层的任何后端（当前是
> `Cuda`），如果没有显式 override，标记就会强制编译错误，因此一个正确但慢的默认实现
> 绝不会在 CUDA 上悄悄发货。bring-up 后端（`Cpu`、未来的 `Metal`）不在这个层里，
> 可以自由继承默认实现。

---

## 2. 模块 / crate 树

### 之前（当前）

```text
infer-worker/src/
├── domain/ { ports/{device.rs, op_ports.rs}, model.rs, batch.rs (BatchPlan<D>), types.rs (closed DataType) }
├── application/ { model_runner/, decode_engine.rs (kv_len += 1), cuda_graph_runner.rs,
│                  forward_workspace.rs (dense-only view god-struct), serve_loop.rs, ... }
├── infrastructure/ { cuda/ (CudaConfig pub fields, thread-local stream), cpu/, transport/, io/ }
├── models/ { llama3.rs (cross-layer norm fusion @213-223), qwen3.rs, diffusion/ }
└── bin/worker_main.rs (ad-hoc `match model_type`, hardcodes <bf16, Cuda>)
```

### 之后（v2）

```text
infer-worker/src/
├── domain/                          # 纯净、后端无关
│   ├── tensor.rs                    # Tensor<T,D: Device> + bitcast/view（不变量 2, 8）
│   ├── dtype/
│   │   ├── mod.rs                   # 开放 dtype：trait Dtype、DTypeId 注册表；Fp8E4m3/Fp8E5m2（不变量 8）
│   │   └── quant.rs                 # QuantScheme {granularity,symmetry,packing,group}（不变量 8）
│   ├── exec.rs                      # Device(+memory)、ExecScope、Stream、ActiveGuard、StepCtx、
│   │                                #   TopologyShape、Rank、QuantTier、MaskHandle（不变量 2,5,6,11）
│   ├── ports/
│   │   ├── math_ops.rs             # trait MathOps: Device —— 可移植底座（约 15 个 op）（不变量 3,4）
│   │   ├── fused_ops.rs            # trait FusedOps: MathOps —— 默认组合，可覆写（不变量 3,7）
│   │   ├── diffusion_ops.rs        # trait DiffusionOps: MathOps —— 融合分支家族（不变量 3,12）
│   │   ├── collective.rs           # trait CollectiveOps: Device —— 单 rank 时无操作（不变量 6,10）
│   │   ├── sampler.rs              # trait Sampler<T,D> —— probs + AcceptReject；不是后端能力（不变量 10）
│   │   ├── backend.rs             # LlmBackend / DiffusionBackend / Backend 的 blanket 别名（不变量 12）
│   │   └── error.rs              # OpError::Unsupported{backend,op}；#[must_override] 重新导出
│   ├── component.rs               # trait Component、StageKind、Hidden、LayerRange、LayerWeights（不变量 7）
│   ├── model.rs                   # trait DecoderModel：embed/decode_layers(range)/finalize（不变量 7）
│   ├── kv.rs                      # PagedKvPool + KvView + KvEdit（不变量 9）
│   └── plan.rs                    # BatchPlan（不带参数）、BatchKind、StepRequest、StepOutput（不变量 4）
├── components/                     # 可复用的具体阶段，每个组件拥有自己的私有 scratch
│   ├── embed.rs · norm.rs · attention.rs · lm_head.rs · linear.rs · quant_linear.rs
│   ├── ffn_dense.rs               # DenseFfn：impl Component
│   ├── ffn_moe.rs                 # MoeFfn：Router + 每个 expert 的权重 + grouped GEMM（MoE 轴）
│   └── decoder_block.rs          # DecoderBlock<T,D,F: Component> —— 持有自己的 norm（不融合）
├── models/
│   ├── llama3.rs · qwen3.rs       # 组装阶段列表；这里保持跨层 norm 不融合
│   └── diffusion/                 # 复用 DiffusionOps + ExecScope
├── application/
│   ├── runtime.rs                 # Runtime<T,D,M>：embed/decode_layers/finalize；单一 ExecScope
│   ├── hosting.rs                 # ModelHost：一个 runtime 中承载 1..N 个模型（spec, PP）
│   ├── exec_scope.rs            # RAII 作用域供给、fork()/record_event/wait_event（保留接口）
│   ├── sampler_stack.rs        # GreedySampler / ChainSampler + verify；选作 Box<dyn Sampler>
│   ├── spec_runtime.rs         # draft+target 共驻，线性链（tree 保留）
│   ├── decode_engine.rs        # 提交由 StepOutput.accepted[seq] 驱动（不变量 9）
│   ├── cuda_graph_runner.rs · serve_loop.rs · worker_scheduler.rs · kv_relief.rs · worker_state.rs
├── infrastructure/
│   ├── cuda/   # 实现 MathOps；覆写 FusedOps/DiffusionOps 的热点分支；CudaScope；CollectiveOps（NCCL 以后）
│   ├── cpu/    # 只实现 MathOps —— 继承 FusedOps 默认实现；CollectiveOps 无操作；Scope 无操作
│   ├── <newbackend>/ # 实现 MathOps（约 15）+ Unsupported；自有 Scope（Metal command buffer / Vulkan queue）
│   ├── transport/ · io/
└── bin/worker_main.rs            # 对已发货的（backend×dtype×arch）元组做宏生成 match
```

---

## 3. 核心抽象

下面所有草图彼此一致：相同的泛型参数、**相同的 scope 传递规则**
（MathOps 接收 `scope: &<Self as Device>::Scope`；fused ops + sampler 接收
 `ctx: &StepCtx<'_, Self>`，其中带着 scope + plan + mask）、相同的 `Tensor<T, D: Device>`
 约束，以及所有 capability trait 都统一使用不带 `self` 的关联函数接收者。

### 3.1 执行与设备接缝 — `domain/exec.rs`

```rust
use std::fmt::Debug;
use std::ptr::NonNull;
use crate::domain::ports::error::OpResult;

/// 这个进程内某个物理设备的稳定身份（CUDA ordinal；CPU 为 0）。
/// 分配以它为键；active guard 会据此驱动 `cudaSetDevice`。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId(pub i32);

/// 一个计算目标（Cuda / Cpu / 未来的 ROCm / Metal / Vulkan）以及它的内存表面。
///
/// `Device` 是唯一的基础约束（不变量 2）：它把旧的 `MemoryPort` 一并折叠进来，
/// 因此 `Tensor<T, D: Device>` 和任何以 `Device` 约束的持有 tensor 的类型都能组合，
/// 不需要第二个内存 trait。所有 capability trait 都是 `: Device`。
pub trait Device: Clone + Send + Sync + Debug + 'static {
    /// 这个设备对应的具体执行作用域类型。`ExecScope` 是对它的约束，
    /// 绝不写成 `ExecScope<Self>`（不变量 5）。CUDA：stream + handles + workspace + rank。
    /// CPU：一个 ZST 的无操作 scope。Metal：一个 command buffer。Vulkan：一个 queue + pool。
    type Scope: ExecScope<Device = Self>;

    fn device_id(&self) -> DeviceId;
    fn name(&self) -> &'static str;

    // ── 内存表面（从旧的 MemoryPort 折叠进来） ──
    /// 在这个设备上分配 `size` 个清零字节（以设备为键、与 scope 无关）。
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>>;
    /// # Safety: `ptr`/`size` 必须对应先前仍然存活的 `alloc_bytes`。
    unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize);
    /// 在 `scope` 的 stream 上同步执行 H2D 拷贝，然后同步该 stream。
    /// # Safety: device/host 指针都必须覆盖至少 `size` 字节。
    unsafe fn upload(&self, scope: &Self::Scope, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;
    /// 在 `scope` 的 stream 上异步执行 H2D 拷贝；不做同步（适合 graph capture）。
    /// # Safety: 同 `upload`。
    unsafe fn upload_async(&self, scope: &Self::Scope, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;
    /// 在 `scope` 的 stream 上同步执行 D2H 拷贝。
    /// # Safety: 指针都必须覆盖至少 `size` 字节。
    unsafe fn download(&self, scope: &Self::Scope, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()>;
    /// 在 `scope` 的 stream 上执行设备内非重叠拷贝。
    /// # Safety: 设备指针互不重叠，且都必须覆盖至少 `size` 字节。
    unsafe fn copy_device_to_device(&self, scope: &Self::Scope, dst: NonNull<u8>, src: NonNull<u8>, size: usize) -> OpResult<()>;
}

/// 主机可寻址内存的标记（支持 `Tensor::as_slice`）。仅 CPU 使用。
pub trait HostDevice: Device {}

/// 统一的执行上下文值（不变量 5, 11）。它持有设备句柄 + 当前 stream + GEMM
/// workspace + 本 scope 的 `Rank` + 当前 `QuantTier`，全部都是**字段**（绝不做成类型参数）。
/// 它替代泄漏出来的 `CudaConfig` 公有字段，以及 diffusion 的 thread-local。必须**显式**传递。
pub trait ExecScope: Send + Sync + Sized + 'static {
    type Device: Device<Scope = Self>;
    /// 后端原生的有序 lane（CUDA stream / Metal command buffer / Vulkan queue / CPU `()`）。
    type Stream: Stream;

    /// 激活这个 scope 的设备，并在 guard 生命周期结束时恢复（RAII 版 `cudaSetDevice`）。
    /// 这是热路径上调用 `cudaSetDevice` 的**唯一**位置，因此多设备正确性是结构性保证
    /// （不变量 5）。`!Send` guard：一个 scope 的激活只会存在于一个线程上。
    fn enter(&self) -> ActiveGuard<'_, Self::Device>;

    /// 唯一的 stream 访问器（替换掉两个旧访问器）。
    fn stream(&self) -> &Self::Stream;
    /// 该 scope 的 mesh 位置（Copy 字段）。`CollectiveOps` 会读取它；`Rank::SINGLE` 是无操作路径。
    fn rank(&self) -> Rank;
    /// mesh 形状（每个轴的 rank/size，Copy 字段，不变量 11）。scope 上不加泛型。
    fn topology(&self) -> TopologyShape;
    /// 当前激活的 KV/weight quant tier（保留位）。默认是 `QuantTier::None`。
    fn quant_tier(&self) -> QuantTier;
    /// 这个 scope 自己持有的 scratch GEMM/reduction workspace（取代泄漏的 `CudaConfig::workspace`）。
    fn workspace(&self) -> &Workspace<Self::Device>;
    /// 阻塞直到该 scope 的 stream 排空。CPU：无操作。
    fn synchronize(&self) -> OpResult<()>;

    // ── 保留的多流 / 跨阶段接缝（今天仍然是单流；见 §6） ──
    // type Event: Event;
    // fn fork(&self) -> OpResult<Self>;
    // fn record_event(&self) -> OpResult<Self::Event>;
    // fn wait_event(&self, ev: &Self::Event) -> OpResult<()>;
}

/// 后端原生的有序 lane 标记（op 代码永远不写具体类型名）。
pub trait Stream: Send + Sync + 'static {}

/// 来自 `ExecScope::enter()` 的 RAII guard。存活期间，scope 的设备处于 current 状态。
/// 故意是 `!Send`。
pub struct ActiveGuard<'a, D: Device> {
    _scope: &'a D::Scope,
    _prev_device: DeviceId,
    _not_send: core::marker::PhantomData<*const ()>,
}
// Drop: restore `cudaSetDevice(_prev_device)`.

/// 由 scope 持有的 scratch arena（取代泄漏的 `CudaConfig::workspace`）。
/// 各组件从这里再分配自己的私有 scratch（不变量 7）；不再有人手工构造原始 GEMM scratch。
pub struct Workspace<D: Device> { _ptr: Option<NonNull<u8>>, _size: usize, _d: core::marker::PhantomData<D> }

/// 跨阶段、按 STEP 传递的 carrier（不是在 ops 之间传递）。它持有 scope + 未参数化的
/// `&BatchPlan`。可变执行状态都放在 scope 后面，因此 `StepCtx` 不会变成 mutable，
/// 也不会在 TP/PP/spec 到来时增加参数。tree/Medusa mask 作为一个保留字段
/// （`MaskHandle`）传递，而不是原始指针（不变量 11；见 §6）。
pub struct StepCtx<'a, D: Device> {
    scope: &'a D::Scope,
    plan: &'a crate::domain::plan::BatchPlan,
    _marker: core::marker::PhantomData<D>,
}

impl<'a, D: Device> StepCtx<'a, D> {
    pub fn new(scope: &'a D::Scope, plan: &'a crate::domain::plan::BatchPlan) -> Self;
    /// 当前 scope（fused ops 会先调用 `ctx.scope().enter()`，然后把 `ctx.scope()` 传给 MathOps）。
    pub fn scope(&self) -> &D::Scope;
    /// 纯粹、后端无关的 batch 元数据（不变量 4：永远不是 `BatchPlan<D>`）。
    pub fn plan(&self) -> &crate::domain::plan::BatchPlan;
}

/// scope 在 mesh 中的位置——作为 scope 上的 Copy 字段存在（不变量 11），供 collective 读取。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rank {
    pub tp_rank: usize, pub pp_rank: usize, pub dp_rank: usize, pub node_rank: usize, pub world_rank: usize,
}
impl Rank { pub const SINGLE: Rank = Rank { tp_rank: 0, pp_rank: 0, dp_rank: 0, node_rank: 0, world_rank: 0 }; }

/// 一个并行轴的 (rank, size)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RankPair { pub rank: usize, pub size: usize }

/// mesh 形状：只包含 ranks/sizes（Copy）。scope 上携带的 SINGLE 来源（不变量 11）。
/// communicator 句柄不在这里——它们放在后端的 `CollectiveOps` 实现里，通过关联函数访问，
/// 因此 scope 永远不会增加一个 `<C>` 泛型。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TopologyShape { pub tp: RankPair, pub pp: RankPair, pub dp: RankPair, pub node: RankPair }
impl TopologyShape {
    pub const SINGLE: TopologyShape = TopologyShape {
        tp: RankPair { rank: 0, size: 1 }, pp: RankPair { rank: 0, size: 1 },
        dp: RankPair { rank: 0, size: 1 }, node: RankPair { rank: 0, size: 1 },
    };
    /// 由保留的协议字段 `LoadModel{tp_rank,tp_size,pp_rank,pp_size}` 构造。
    pub fn from_load_model(load: &infer_protocol::scheduler_to_worker_control::LoadModel) -> Self;
    pub fn world_size(&self) -> usize; // tp.size * pp.size * dp.size * node.size
    pub fn rank_in(&self, axis: CommAxis) -> usize;
    pub fn group_size(&self, axis: CommAxis) -> usize;
    pub fn is_pp_first(&self) -> bool;
    pub fn is_pp_last(&self) -> bool;
}

/// scope 携带的保留量化层级。今天是 `None`；新增层级只加新分支，不加新方法。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantTier { None /*, Fp8Kv, Int4Kv (reserved) */ }

/// 指向设备驻留 attention mask 的后端无关句柄（tree/Medusa，保留）。
/// 它是注册表索引，不是原始指针——在 kernel site 通过 scope 解析（见 §6）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskHandle(pub u64);

/// `must_override` 属性宏（crate `infer-worker-macros`）的重新导出。
pub use infer_worker_macros::must_override;
```

> 上面引用的 `CommAxis` 枚举定义在 §3.3（collectives），并在这里复用。

### 3.2 操作端口 — `domain/ports/{error,math_ops,fused_ops,diffusion_ops,backend}.rs`

```rust
// ── error.rs ──
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error("shape error: {0}")]                                  Shape(String),
    #[error("not contiguous")]                                    NotContiguous,
    /// 允许的 bring-up 阀门：后端拒绝一个它还没有实现 kernel 的 op。
    /// 静态字符串 → 无分配 + 易检索。
    #[error("unsupported op '{op}' on backend '{backend}'")]      Unsupported { backend: &'static str, op: &'static str },
    #[error("kernel failed: {0}")]                                Kernel(String),
    #[error("shutdown requested")]                                Shutdown,
}
impl OpError { pub fn unsupported(backend: &'static str, op: &'static str) -> Self; }
pub type OpResult<T> = std::result::Result<T, OpError>;
```

```rust
// ── math_ops.rs —— 可移植底座（不变量 3, 4, 6） ──
use crate::domain::exec::{Device, ExecScope};
use crate::domain::dtype::{Dtype, quant::QuantScheme};
use crate::domain::tensor::{Tensor, Shape};
use crate::domain::ports::error::OpResult;

/// 新后端**唯一**必须实现的 surface。每个方法都与 dtype 泛型无关、纯 tensor 输入/输出，
/// 并且是可移植的。活跃的 stream/device 通过**显式的**
/// `scope: &<Self as Device>::Scope` 取得（不变量 5）—— 没有 thread-local。签名里不放
/// `BatchPlan`、不放分页 KV 地址、不放任何 device scratch（不变量 4）。所有方法都是关联函数
/// （不变量 6）。
pub trait MathOps: Device {
    /// 分配一个连续的清零 tensor。 // 默认：Tensor::<T,Self>::zeros(shape, device)
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>>;

    // ── 元素级 ──
    fn add<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    fn add_inplace<T: Dtype>(scope: &Self::Scope, dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()>;
    fn ewise_mul<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    fn scalar_mul_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()>;
    fn broadcast_mul_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, scale: &Tensor<T, Self>) -> OpResult<()>;
    fn broadcast_add_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, bias: &Tensor<T, Self>) -> OpResult<()>;

    // ── 线性代数 ──
    /// Dense matmul `[M,K] × [N,K]^T → [M,N]`，同 dtype。
    fn matmul<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 量化 matmul。`scheme`（不变量 8）完整描述 granularity/symmetry/packing/group —— 它
    /// 取代了裸露的 `group_size`。`zeros` 仅当 `scheme.symmetry == Asymmetric` 时为 `Some`。
    /// 后端按 `scheme.packing` 匹配，并对未实现的 packing 返回 `Unsupported`。新的量化族 =
    /// 在这里消费一个新的 `Packing` 分支，绝不是新增方法。
    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<A, Self>, weight: &Tensor<W, Self>, output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>, zeros: Option<&Tensor<W, Self>>, scheme: &QuantScheme,
    ) -> OpResult<()>;

    // ── 归一化（可移植原语——不融合，不变量 7） ──
    fn rmsnorm<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>, eps: f32) -> OpResult<()>;
    fn rmsnorm_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32) -> OpResult<()>;

    // ── 激活 ──
    fn silu_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>) -> OpResult<()>;
    fn softmax<T: Dtype>(scope: &Self::Scope, input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

    // ── RoPE（可移植；分页 scatter 不在这里） ──
    fn rope_inplace<T: Dtype>(
        scope: &Self::Scope, q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>, cos: &Tensor<T, Self>, positions: &Tensor<i32, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize,
    ) -> OpResult<()>;

    // ── dense attention primitive（无 KV pool、无 plan） ──
    /// 对连续 materialized 的 q/k/v 做 SDPA，并可选加性 mask。可移植 attention 底座：
    /// `FusedOps::attention_paged` 会先 gather paged K/V，再默认组合出它。支持 GQA。
    fn sdpa<T: Dtype>(
        scope: &Self::Scope, q: &Tensor<T, Self>, k: &Tensor<T, Self>, v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>, mask: Option<&Tensor<T, Self>>,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()>;

    // ── embedding / shape / dtype ──
    fn embedding<T: Dtype>(scope: &Self::Scope, table: &Tensor<T, Self>, indices: &Tensor<i32, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;
    fn split_cols<T: Dtype>(scope: &Self::Scope, src: &Tensor<T, Self>, dst: &mut Tensor<T, Self>, rows: usize, total_cols: usize, col_offset: usize, dst_cols: usize) -> OpResult<()>;
    fn concat_seq<T: Dtype>(scope: &Self::Scope, a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 同设备上的数值 dtype 转换。覆盖 fp8 ↔ bf16/f16/f32（fp8 也是普通 Dtype）。
    fn cast<S: Dtype, T: Dtype>(scope: &Self::Scope, src: &Tensor<S, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 不复制地重新解释字节（不变量 8）：在同一块存储上为 `Tensor<S>` 建立 `Tensor<T>` 视图。
    /// 取代 int4-as-i32。只有当 `S::SIZE_BYTES * src.numel()` 能被 `T::SIZE_BYTES` 整除时才允许。
    /// 纯视图运算（O(1) Arc bump）——可移植默认实现，绝不覆写。
    fn bitcast<S: Dtype, T: Dtype>(src: &Tensor<S, Self>, new_shape: Shape) -> OpResult<Tensor<T, Self>>;
}
```

```rust
// ── fused_ops.rs —— LLM 上限（不变量 3, 4, 7） ──
use crate::domain::exec::StepCtx;
use crate::domain::kv::{KvView, LayerKv};
use crate::domain::plan::BatchPlan;

/// 编排 / 融合 LLM ops。每个方法都有完全由 `MathOps` 组合出来的默认体（不变量 3）：
/// 只实现 `MathOps` 的后端也满足 `FusedOps`，并能正确运行 LLM（只是慢一些）。CUDA 负责
/// 覆写热点分支。编排物理（分页 KV、未参数化的 `&BatchPlan`、mask 模式）都在这里，
/// 通过 `ctx: &StepCtx` 访问（它携带 scope + plan）。所有方法都是关联函数（不变量 6）。
pub trait FusedOps: MathOps {
    /// 融合的 `residual += input; output = rmsnorm(residual, weight, eps)`。
    /// 这是把旧的跨层 norm fusion **显式**重新拿回来的地方（不变量 7）。
    /// 默认：add_inplace(residual,input); rmsnorm(residual,weight,output,eps)
    fn fused_add_rmsnorm<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        output: &mut Tensor<T, Self>, residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32,
    ) -> OpResult<()>;

    /// 打包的 SwiGLU `gate_up[rows,2*inter] → out[rows,inter]`。
    /// 默认：split_cols gate&up; silu_inplace(gate); ewise_mul(gate,up,out)
    fn swiglu_packed<T: Dtype>(ctx: &StepCtx<'_, Self>, gate_up: &Tensor<T, Self>, out: &mut Tensor<T, Self>, rows: usize, inter: usize) -> OpResult<()>;

    /// 将 fused `[num_tokens, qkv_dim]` 拆成 Q/K/V。默认：三个 split_cols。
    fn split_qkv<T: Dtype>(ctx: &StepCtx<'_, Self>, qkv: &Tensor<T, Self>, q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>, v: &mut Tensor<T, Self>, num_tokens: usize, q_dim: usize, kv_dim: usize) -> OpResult<()>;

    /// 在某层的 KV 视图上执行分页 attention。mask 纪律来自 `plan.kind` 的 `MaskMode`
    ///（不是裸露的 `is_causal: i32`）；scratch 来自 `ctx.scope().workspace()`。
    /// `kv` 把 pool 切片 + device 索引张量绑在一起，因此 `BatchPlan` 保持不带参数（不变量 4）。
    /// 默认：gather paged K/V → contiguous，按 plan 构造 additive mask，然后调用 MathOps::sdpa。
    #[must_override(tier = "perf")]
    fn attention_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &Tensor<T, Self>, kv: &KvView<'_, T, Self>, output: &mut Tensor<T, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()>;

    /// 将 K/V 行 scatter 到某层的分页 pool 中。索引张量来自 `kv.index`；签名只描述分页几何。
    /// 支持按位置覆盖写。`LayerKv` 写句柄是 `&mut`。
    /// 默认：逐 token 的 block-id 算术 + `Device::copy_device_to_device`。
    #[must_override(tier = "perf")]
    fn scatter_kv_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        k_src: &Tensor<T, Self>, v_src: &Tensor<T, Self>,
        layer: &mut LayerKv<'_, T, Self>, kv_dim: usize,
    ) -> OpResult<()>;

    /// 融合的 Q/K-RMSNorm + RoPE + paged scatter（Qwen 路径）。对 fused = composed 的证明路径。
    /// 默认：对 Q/K 的 head-view 先做可选 rmsnorm_inplace；然后 rope_inplace；最后 scatter_kv_paged。
    #[must_override(tier = "perf")]
    fn qkv_norm_rope_scatter<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>, v: &Tensor<T, Self>,
        q_weight: Option<&Tensor<T, Self>>, k_weight: Option<&Tensor<T, Self>>, q_eps: f32, k_eps: f32,
        sin: &Tensor<T, Self>, cos: &Tensor<T, Self>, positions: &Tensor<i32, Self>,
        layer: &mut LayerKv<'_, T, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize, kv_dim: usize,
    ) -> OpResult<()>;

    /// MoE 的 grouped expert GEMM。`expert_offsets[num_experts+1]` 定义 token 分组（按 expert 排序后）；
    /// `weights` 是堆叠的 expert tensor；可选 `scheme` 支持量化 expert。
    /// 默认：遍历 experts，按 offset 切 token rows，对每组做 matmul/matmul_quant（dropless，单 GPU）。
    /// expert-parallel 以后会在 `MoeFfn` 里加 `CollectiveOps::all_to_all`，**不在这里**。
    #[must_override(tier = "perf")]
    fn grouped_expert_gemm<A: Dtype, W: Dtype, O: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<A, Self>, weights: &Tensor<W, Self>, output: &mut Tensor<O, Self>,
        expert_offsets: &Tensor<i32, Self>,
        scales: Option<&Tensor<A, Self>>, zeros: Option<&Tensor<W, Self>>, scheme: Option<&QuantScheme>,
    ) -> OpResult<()>;
}
```

> **ceiling 里没有 `fused_decode_layer`。** 根据评审意见，一个带 `&dyn LayerWeights` 的整层
> megakernel 会把 `dyn` 放到 op 路径上（违反不变量 1），而且对 MoE 也不是增量式的。
> 跨层 fusion 通过 `fused_add_rmsnorm` + `qkv_norm_rope_scatter` + `swiglu_packed` 恢复。
> 如果以后确实需要真正的整层 megakernel，那它应该是 `DecoderBlock<F>` 上的一个**具体**
> 方法（静态的，知道自己的 `F`），而不是一个 dyn-weights trait 方法。

```rust
// ── backend.rs —— 能力别名（不变量 10, 12） ──
/// DECODER 路径使用的干净约束。它不包含 DiffusionOps（不变量 12），也不包含 Sampler
///（不变量 10：Sampler 是策略，不是后端能力）。通过 blanket impl 提供，零后端样板代码。
pub trait LlmBackend: FusedOps + CollectiveOps {}
impl<D: FusedOps + CollectiveOps> LlmBackend for D {}

/// DIFFUSION 模型使用的约束。
pub trait DiffusionBackend: DiffusionOps + CollectiveOps {}
impl<D: DiffusionOps + CollectiveOps> DiffusionBackend for D {}

/// 适用于“什么都做”的后端的便利上层别名（例如统一 CUDA 构建）。
pub trait Backend: LlmBackend + DiffusionBackend {}
impl<D: LlmBackend + DiffusionBackend> Backend for D {}
```

### 3.3 Collectives 与拓扑 — `domain/ports/collective.rs`

```rust
use crate::domain::exec::{Device, ExecScope};
use crate::domain::dtype::Dtype;
use crate::domain::tensor::Tensor;
use crate::domain::ports::error::OpResult;

/// collective 所运行的逻辑 mesh 维度。故意是闭合的：worker 将承载的每个并行轴都在这里。
/// 新增一个轴 = 新增一个 variant，而不是新增一个 trait 方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum CommAxis { Tp, Pp, Dp, Ep }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum ReduceOp { Sum, Max, Min, Avg }

/// 进程 mesh 上的通信能力。它是 `: Device` 的侧车（共享 tensor 主干），但**不**属于
/// 可移植 math 底座。所有方法都是关联函数（不变量 6），显式接收 `scope`（rank 从
/// `scope.rank()`/`scope.topology()` 读取，而不是作为参数——不变量 11）。
/// communicator 句柄通过 concrete backend 上的 `comm(axis)` 访问；它们**不是** scope 上的
/// 泛型字段，因此不变量 11（scope 绝不增加第二个泛型）成立。
///
/// 单 rank 实现会把每个方法都变成恒等无操作（all_reduce 原样返回输入，all_gather 复制本地分片，
/// send/recv 不可达）→ 今天的路径保持字节级一致。
pub trait CollectiveOps: Device {
    /// 一个进程组句柄（CUDA：`ncclComm_t` newtype；CPU/single-rank：ZST）。
    type Comm: Send + Sync;

    /// 借用这个后端在 `axis` 上的 communicator（当该轴 size==1 时返回 None）。
    /// 具体类型，绝不会出现在 op 签名里。
    fn comm(scope: &Self::Scope, axis: CommAxis) -> Option<&Self::Comm>;

    /// 在 `axis` 上做原地 all-reduce（TP 在 row-parallel 接缝处的主力）。
    fn all_reduce<T: Dtype>(scope: &Self::Scope, axis: CommAxis, op: ReduceOp, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 将每个 rank 的 `shard` 按 `dim` gather 到 `out` 中，每个 rank 都得到一份。
    fn all_gather<T: Dtype>(scope: &Self::Scope, axis: CommAxis, dim: usize, shard: &Tensor<T, Self>, out: &mut Tensor<T, Self>) -> OpResult<()>;
    /// reduce + scatter，使每个 rank 保留自己的 `1/group_size` 切片。
    fn reduce_scatter<T: Dtype>(scope: &Self::Scope, axis: CommAxis, op: ReduceOp, dim: usize, buf: &Tensor<T, Self>, out: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 将 `buf` 从 `root`（`axis` 内索引）广播到所有成员。
    fn broadcast<T: Dtype>(scope: &Self::Scope, axis: CommAxis, root: usize, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// 在 PP stage 之间发送/接收 `Hidden`（`peer` 是 `axis` 内索引）。
    fn send<T: Dtype>(scope: &Self::Scope, axis: CommAxis, peer: usize, buf: &Tensor<T, Self>) -> OpResult<()>;
    fn recv<T: Dtype>(scope: &Self::Scope, axis: CommAxis, peer: usize, buf: &mut Tensor<T, Self>) -> OpResult<()>;
    /// all-to-all 交换。保留给 expert-parallel MoE dispatch/combine（见 §6）。
    fn all_to_all<T: Dtype>(scope: &Self::Scope, axis: CommAxis, send_chunks: &[Tensor<T, Self>], recv_chunks: &mut [Tensor<T, Self>]) -> OpResult<()>;
    /// `axis` 上的 mesh barrier（调试 / drain）。单 rank：无操作。
    fn barrier(scope: &Self::Scope, axis: CommAxis) -> OpResult<()>;
}

/// 一个权重 tensor 如何在 TP 组上分片（load-time 关注点；TP 不改变 runtime 签名）。
/// 新的分片方式 = 新 variant，而不是新的 loader 方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardSpec {
    Replicated,
    ColumnParallel { dim: usize },  // q/k/v proj, gate/up
    RowParallel    { dim: usize },  // o_proj, down_proj → 接缝处 all_reduce(Sum)
    VocabParallel  { dim: usize },  // embedding / lm_head（采样时保留 gather）
}

/// load-time 切片 hook：Component 只拉取自己的 TP shard。`tp_size == 1` 时无论 `spec` 是什么，
/// 都会返回完整 tensor，因此现有模型会保持字节级一致加载。
pub trait ShardedLoad: Device {
    fn load_shard<T: Dtype>(&self, rank: crate::domain::exec::Rank, name: &str, spec: ShardSpec) -> OpResult<Tensor<T, Self>>;
}
```

### 3.4 模型与层 — `domain/{component,model}.rs`, `components/*`

```rust
// ── component.rs ──
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::{backend::LlmBackend, error::OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::dtype::Dtype;

/// 一等的跨阶段传递值（不变量 7）。替代 dense-only 的 `LlmForwardWorkspace`。
/// 只携带残差流 `[num_tokens, dim]`。没有中心化的 scratch role enum——每个 `Component`
/// 都持有自己的私有 scratch（从 scope `Workspace` 中再分配）。
/// CUDA graph capture 的地址稳定性由 Runtime 在创建时一次性分配 `Hidden` 槽位并以 `&mut`
/// 传递来保证（见 §3.8）；组件不会在每一步重新分配它。
pub struct Hidden<T: Dtype, D: LlmBackend> {
    /// 残差流 `[num_tokens, dim]`：每个阶段读写的唯一值。
    pub stream: Tensor<T, D>,
}
impl<T: Dtype, D: LlmBackend> Hidden<T, D> {
    pub fn num_tokens(&self) -> usize;
    /// EAGLE / spec / PP hidden-state tap，位于两个 stage 之间（默认是浅层）。
    pub fn tap_stream(&self, deep: bool) -> OpResult<Tensor<T, D>>;
}

/// 组件是什么类型的 stage——驱动 PP 切分决策 + tap introspection。
/// `#[non_exhaustive]`：未来新增一种 kind（例如 cross-attn）只需要加一个 arm。
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum StageKind { Embed, Norm, Attention, Ffn /* dense 或 MoE —— 替换对外不可见 */, LmHead, DecoderBlock }

/// decoder layer 的包含-排除范围，用于切片 forward（PP / EAGLE / spec）。
#[derive(Clone, Copy, Debug)]
pub struct LayerRange { pub start: usize, pub end: usize }
impl LayerRange {
    pub fn all(num_layers: usize) -> Self;
    pub fn single(i: usize) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    /// 将 `num_layers` 按 `pp_size` 均分后的这个 `pp_rank` 区间（唯一的 PP 专用计算）。
    pub fn for_pp_rank(pp_rank: usize, pp_size: usize, num_layers: usize) -> Self;
}

/// 原子计算砖块。它在后端 `D` 上把 `Hidden -> Hidden` 映射起来，并接收每步的 `StepCtx`
/// （其中带着 scope + plan）。`&self`（推理是纯的）。TP collective 接缝固定在
/// `Attention`/`Ffn` 实现的内部，而不是这个签名里。`kv` 仅在带 attention 的 stage 才是
/// `Some`。这个视图是单个 `KvView`，其可变 layer 访问在内部通过 borrow-split 来支持
/// scatter-then-attend（不变量 9；见 §3.5）。
pub trait Component<T: Dtype, D: LlmBackend> {
    fn kind(&self) -> StageKind;
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
}
```

```rust
// ── model.rs ──
use crate::domain::component::{Hidden, LayerRange, StageKind};

/// 几何描述。为了让 dense 模型保持字节级一致，MoE 维度默认扩宽为 0。
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub dim: usize, pub q_dim: usize, pub kv_dim: usize, pub qkv_dim: usize,
    pub intermediate_size: usize, pub vocab_size: usize,
    pub head_num: usize, pub head_dim: usize, pub kv_head_num: usize, pub num_layers: usize,
    // ── MoE 扩宽：dense 情况下全部默认为 0（增量式） ──
    pub num_experts: usize, pub experts_per_tok: usize, pub moe_intermediate_size: usize, pub num_shared_experts: usize,
}
impl Default for ModelDims { fn default() -> Self; }
impl ModelDims {
    pub fn validate(&self) -> OpResult<()>;
    pub fn is_moe(&self) -> bool; // num_experts > 0
}

/// `finalize` 返回的 logits: `[num_sampled_rows, vocab]`。
pub struct Logits<T: Dtype, D: LlmBackend>(pub Tensor<T, D>);

/// `finalize` 要投影哪些 token rows（避免隐含的“所有 rows”假设）。
pub enum SampleRows<'a> { All, LastPerSeq, Explicit(&'a [i32]) }

/// 可切片的 decoder model。没有单体式 kernel-fused 契约。`forward` 只作为一个默认提供的
/// default，把三个接缝串起来，因此现有调用点仍能走一把式路径，而 PP/EAGLE/spec 直接
/// 使用这些切片。约束在 `LlmBackend` 上（不变量 12），这是唯一干净的约束。
pub trait DecoderModel<T: Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    /// 执行顺序中的 stage 描述（供 PP planner 使用；不执行）。
    fn stages(&self) -> &[StageKind];

    /// 接缝 1 —— token ids → 初始 `Hidden`。写入 runtime 预先提供、地址稳定的 `hidden` 槽位
    ///（不分配 —— 保持 CUDA graph 的指针稳定性）。
    fn embed(&self, input_ids: &Tensor<i32, D>, hidden: &mut Hidden<T, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;

    /// 接缝 2 —— 原地运行 `[range.start, range.end)` 之间的 layers。这是 PP/EAGLE/spec 的切片。
    /// 跨层 norm 不融合：每个 block 都应用自己的 input+post-attn norms；融合只在
    /// `FusedOps::fused_add_rmsnorm` 内部重新拿回，绝不作为 model 级契约。
    fn decode_layers(&self, range: LayerRange, hidden: &mut Hidden<T, D>, kv: &mut KvView<'_, T, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;

    /// 接缝 3 —— final norm + LM head → logits（只在最后一个 PP stage）。
    /// `rows` 选择要投影的 token。
    fn finalize(&self, hidden: &Hidden<T, D>, rows: SampleRows<'_>, ctx: &StepCtx<'_, D>) -> OpResult<Logits<T, D>>;

    /// 提供的单体 default = embed → decode_layers(all) → finalize。单 rank dense 路径仍然是
    /// 与今天的 `forward` 字节级一致。
    fn forward(&self, input_ids: &Tensor<i32, D>, hidden: &mut Hidden<T, D>, kv: &mut KvView<'_, T, D>, rows: SampleRows<'_>, ctx: &StepCtx<'_, D>) -> OpResult<Logits<T, D>>;
}
```

```rust
// ── components/ffn_dense.rs & ffn_moe.rs —— dense↔MoE 是一个 COMPONENT 替换 ──
use crate::components::linear::Linear;

/// Dense SwiGLU FFN（当前的 Llama3/Qwen3 路径）。它自己持有 post-attention norm
/// （跨层 fusion 已经消失）。私有 scratch 从 scope workspace 再分配。
pub struct DenseFfn<T: Dtype, D: LlmBackend> { pub gate_up_proj: Linear<T, D>, pub down_proj: Linear<T, D> }
impl<T: Dtype, D: LlmBackend> Component<T, D> for DenseFfn<T, D> {
    fn kind(&self) -> StageKind; // Ffn
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // gate_up = gate_up_proj(h); swiglu_packed; down_proj
}

/// 稀疏 MoE FFN。与 DenseFfn 使用相同的 `Component` 契约 → 把它换进去就是整个 MoE 模型变更。
/// Router top-k + grouped_expert_gemm 是 FusedOps（默认由 MathOps 组合）。expert 中间结果
/// 放在这个组件自己的私有 scratch 中——没有共享 role enum，也没有 workspace 重塑。
pub struct MoeFfn<T: Dtype, D: LlmBackend> {
    pub router: Linear<T, D>,
    pub expert_gate_up: Tensor<T, D>, // [num_experts, 2*moe_inter, dim]
    pub expert_down: Tensor<T, D>,    // [num_experts, dim, moe_inter]
    pub shared: Option<DenseFfn<T, D>>,
    pub experts_per_tok: usize,
}
impl<T: Dtype, D: LlmBackend> Component<T, D> for MoeFfn<T, D> {
    fn kind(&self) -> StageKind; // Ffn（对 runtime 来说，swap 是不可见的）
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // logits=router(h); (ids,w)=topk; 按 expert permute 到私有 scratch; grouped_expert_gemm;
    // unpermute+weight; 如果有 shared 就再跑 shared.run()。无丢弃，固定 per-expert 容量（可 graph 化，见 §6）。
}

// ── components/decoder_block.rs —— FFN 泛型化；norm 在这里持有（不融合） ──
use crate::components::{norm::RmsNorm, attention::Attention};
pub struct DecoderBlock<T: Dtype, D: LlmBackend, F: Component<T, D>> {
    pub input_layernorm: RmsNorm<T, D>,
    pub attention: Attention<T, D>,
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub ffn: F, // DenseFfn 或 MoeFfn —— 在 model build 时选定，静态分发
}
impl<T: Dtype, D: LlmBackend, F: Component<T, D>> Component<T, D> for DecoderBlock<T, D, F> {
    fn kind(&self) -> StageKind; // DecoderBlock（PP 切分单元）
    fn run(&self, hidden: &mut Hidden<T, D>, kv: Option<&mut KvView<'_, T, D>>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    // input_layernorm → attention(scatter then attend on one KvView) → fused_add_rmsnorm → ffn.run
}
```

### 3.5 KV cache 与 batch plan — `domain/{kv,plan}.rs`

```rust
// ── plan.rs —— BatchPlan 不带参数（不变量 4）。唯一权威定义。 ──
use crate::domain::exec::MaskHandle;

/// ragged paged-attention kernel 的 Q tile 大小（必须与 `kBlockM` 一致）。
pub const RAGGED_Q_TILE: i32 = 128;

/// attention mask 纪律——对裸露 `is_causal: i32` 的类型化替代。只有一个 enum；
/// spec 直接复用它（不会并行再来一个 `SpecMaskMode`）。新的 regime = 新 arm + 一个 kernel 分支，
/// 绝不新增方法。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    Full,                              // diffusion / encoder self-attention
    Causal,                           // 标准 autoregressive（今天的 is_causal==1）
    SlidingWindow { window: u32 },    // 保留
    Tree,                             // 保留 tree/Medusa——读取 BatchKind::Spec 的 mask handle
}

/// 这一步是什么 attention 变体。驱动 kernel dispatch + attention component 构造的 mask。
/// `Spec` 携带保留的 tree mask handle（只有这一处，不要也放在 StepCtx 上）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    DecodeOnly,                                       // 每个 seq q_len==1 → Flash-Decoding
    Ragged,                                           // 可变 q_len>=1 → tile-scheduled kernel
    Spec { mask: MaskMode, mask_handle: Option<MaskHandle> }, // 当前是线性链（mask=Causal, handle=None）
}

/// 单步 forward 的计划——纯 host 侧元数据 + schedule，与 backend 无关（不变量 4）。
/// 解决重复定义的决定：device-resident 的 index tensors 不在这里；它们在 `KvIndexTensors`
///（位于设备上、在 scope 内）。`BatchPlan` 只保存 host Vec。runtime 会在每一步把它们上传到
/// `KvIndexTensors`。
#[derive(Debug, Clone)]
pub struct BatchPlan {
    pub kind: BatchKind,
    pub num_tokens: usize,
    pub batch: usize,
    pub q_lens: Vec<i32>,          // 每个 seq 的 q_len；decode 时全为 1（消解“每行一个槽位”）
    pub kv_lens: Vec<i32>,         // 本步之后每个 seq 的 KV 长度
    pub seq_positions: Vec<i32>,   // 每个 seq 写入 cache 的起始行（== 之前的 kv_len）
    pub rope_positions: Vec<i32>,  // 每个 token 的绝对 RoPE 位置
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub total_q_tiles: i32,        // ragged tile 数（DecodeOnly 时为 0）
}
impl BatchPlan {
    pub fn is_decode_only(&self) -> bool;
    /// host 侧前缀和 + ragged tile schedule（所有 backend 都上传同一份 schedule）。
    pub fn plan_ragged_tiles(q_lens: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>); // (cu_q_lens, block2req, block2tile)
}

/// 供调用方传入的 per-seq host 描述（今天的 `SeqStep`，形状不变）。
#[derive(Debug, Clone)]
pub struct SeqStep { pub sequence_id: SeqId, pub input_ids: Vec<i32>, pub positions: Vec<i32>, pub kv_write_start: i32, pub kv_len_after: i32, pub block_table: Vec<u32> }

/// 一个 token 结果，带 probability 信息（修复“只有 argmax、没有概率”的结构问题）。
#[derive(Debug, Clone)]
pub struct SampledToken { pub token_id: i32, pub logprob: f32, pub top_logprobs: Vec<(i32, f32)> }

/// 统一的 runtime → step 接缝（由 `Runtime::step` 消费）。
#[derive(Debug, Clone)]
pub struct StepRequest {
    pub seqs: Vec<SeqStep>,
    pub sampling: Vec<crate::domain::ports::sampler::SamplingParams>, // 转交给 Sampler 策略
    pub stop: StopCriteria,
    pub draft_tokens: Vec<Vec<i32>>, // 只用于 spec verify；否则为空
}

/// 统一的 step 结果——唯一的 commit 来源（不变量 9）。`accepted[i]` 取代 `kv_len += 1`：
/// decode=1，prefill 接受量= q_len，spec verify=1..=K。只有一个元素类型 (`SampledToken`)
/// 携带 logprob——没有并行的 `logprobs` vec。
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub tokens: Vec<Vec<SampledToken>>, // 每个 seq 一个，按 accepted[i] 呈 ragged
    pub accepted: Vec<u32>,
    pub finished: Vec<bool>,
    pub hidden_tap: Option<HiddenTap>, // EAGLE/PP 导出；普通 decode 时为 None（保留但已接线）
}

#[derive(Debug, Clone)]
pub struct StopCriteria { pub eos_ids: Vec<i32>, pub generated_counts: Vec<u32>, pub max_tokens: Vec<u32>, pub ignore_eos: bool }

/// 为 EAGLE/PP 捕获的 hidden state，对 runtime 透明。
pub struct HiddenTap { pub at_layer: usize /* + Hidden stream，dtype/shape 对 transport 擦除 */ }
```

```rust
// ── kv.rs —— PagedKvPool + KvView（单一 view 类型）+ KvEdit（mutation）。pool 持有 per-seq kv_len
//    因此 KvEdit 是自包含的（不变量 9）。KvIndexTensors 被并入 view（依据评审意见）。 ──
use crate::domain::exec::Device;
use crate::domain::dtype::quant::QuantScheme;

/// 保留的 KV-cache 量化层级。`None` 是当前发货形态（full precision，字节级一致）。
/// 新方案复用共享的 `QuantScheme`；不要新增 pool 方法。
#[derive(Debug, Clone)]
pub enum KvQuantTier {
    None,
    PerTensor(QuantScheme),                                  // 现在可用的层级（fp8/int8 KV）
    PerBlock { scheme: QuantScheme /*, per-block scale pool — 保留 */ }, // 保留（§6）
}

/// 单个 transformer layer 的切片。形状是 K 和 V 各自的 `[num_blocks, block_size, kv_dim]`。
pub struct PagedKvLayer<T: Dtype, D: Device> { pub k: Tensor<T, D>, pub v: Tensor<T, D> }

/// 设备驻留的 per-step index tensors（今天的 block_tables/cu_q_lens/... 从 plan 中拆出，
/// 这样 plan 保持后端无关）。由 runtime 在每一步构造，驻留在 scope 上。
pub struct KvIndexTensors<D: Device> {
    pub block_tables: Tensor<i32, D>,   // [batch, max_blocks_per_seq]
    pub cu_q_lens: Tensor<i32, D>,      // [batch+1]
    pub kv_lens: Tensor<i32, D>,        // [batch]
    pub seq_positions: Tensor<i32, D>,  // [batch]
    pub seq_lens_step: Tensor<i32, D>,  // [batch] (== q_len[i])
    pub rope_positions: Tensor<i32, D>, // [num_tokens]
    pub block2req: Tensor<i32, D>,      // ragged schedule（DecodeOnly 时放占位）
    pub block2tile: Tensor<i32, D>,
}

/// worker 持有的 paged pool（不变量 9 的底座）。持有 per-seq `kv_len`。
/// 通过 `KvView` 借出某个 layer-range；通过 `KvEdit` 修改。
pub struct PagedKvPool<T: Dtype, D: Device> {
    pub layers: Vec<PagedKvLayer<T, D>>,
    pub num_blocks: usize, pub block_size: usize, pub kv_dim: usize,
    pub quant: KvQuantTier,                 // 默认 None
    pub seq_kv_len: std::collections::HashMap<SeqId, u32>, // pool-owned per-seq length（不变量 9）
}
impl<T: Dtype, D: Device> PagedKvPool<T, D> {
    pub fn num_layers(&self) -> usize;
    /// 借出 `range` 的 layers + 这一步的设备 index tensors，作为一个 `KvView`。
    /// PP 运行 `decode_layers(range)`，因此 view 也按 range 划分。
    pub fn view<'a>(&'a mut self, range: LayerRange, index: &'a KvIndexTensors<D>) -> KvView<'a, T, D>;
    /// 打开一个 mutation transaction，供 commit 阶段使用。
    pub fn edit<'a>(&'a mut self) -> KvEdit<'a, T, D>;
}

/// 单一的 paged view，供一个 layer-range 使用（同时读写）。持有 `&mut [PagedKvLayer]`；
/// 某个 layer 会为 `scatter_kv_paged` 派生出一个 per-layer `LayerKv`（可变），然后用同一块
/// 存储做 `attention_paged` 读取 —— 在这个单一类型内部通过 borrow split 实现 scatter-then-attend
///（不需要单独的 `KvViewMut`）。
pub struct KvView<'a, T: Dtype, D: Device> {
    pub layers: &'a mut [PagedKvLayer<T, D>],
    pub index: &'a KvIndexTensors<D>,
    pub num_blocks: usize, pub block_size: usize, pub kv_dim: usize,
    pub quant: &'a KvQuantTier,
}
impl<'a, T: Dtype, D: Device> KvView<'a, T, D> {
    /// 单层的可变写入句柄（传给 `scatter_kv_paged`）。
    pub fn layer_mut(&mut self, layer_idx: usize) -> LayerKv<'_, T, D>;
    /// 单层的只读 K/V（在 scatter 完成后，传给 `attention_paged`）。
    pub fn layer(&self, layer_idx: usize) -> (&Tensor<T, D>, &Tensor<T, D>);
}

/// 单层的可变 K/V + indices，用于 scatter 写路径。
pub struct LayerKv<'a, T: Dtype, D: Device> { pub k: &'a mut Tensor<T, D>, pub v: &'a mut Tensor<T, D>, pub index: &'a KvIndexTensors<D> }

/// per-seq KV mutation 计划——是 `seq.kv_len += 1` + one-block push 的结构化替代。
/// 从 `StepOutput.accepted` 构造。pool 拥有 kv_len，因此它是自包含的。
/// `rollback` 不是方法（它就是 `truncate(kv_len - n)`）；只有 chained spec 真需要时再加语法糖（§6）。
pub struct KvEdit<'a, T: Dtype, D: Device> { pub pool: &'a mut PagedKvPool<T, D> }
impl<'a, T: Dtype, D: Device> KvEdit<'a, T, D> {
    /// 提交 `sid` 新写入的 `n` 行（decode 时 n=1；spec 时 n=accepted）。
    /// 只有当 kv_len 越过 block 边界时才分配新的物理 block——按多槽增长，不是按行增长。
    fn append(&mut self, sid: SeqId, n: u32) -> OpResult<()>;
    /// 将 `sid` 截断到精确的 `to` 长度。spec rejection：在写入候选 run 并接受 k 之后，
    /// `truncate(sid, base + k)` 会丢掉被拒绝的尾部。返回释放出来的物理 block。
    fn truncate(&mut self, sid: SeqId, to: u32) -> OpResult<Vec<u32>>;
    /// 一次性应用整步的 accepted counts（`decode_engine` 调用的就是这个）。返回释放块。
    fn apply_step(&mut self, sids: &[SeqId], accepted: &[u32], speculative_len: &[u32]) -> OpResult<Vec<u32>>;
}

/// 稳定的每序列身份（与调度器的 `sequence_id: u64` 对应）。
pub type SeqId = u64;
```

### 3.6 采样 — `domain/ports/sampler.rs`

```rust
use crate::domain::exec::StepCtx;
use crate::domain::tensor::Tensor;
use crate::domain::dtype::Dtype;
use crate::domain::ports::{backend::LlmBackend, error::OpResult};
use crate::domain::plan::SampledToken;

/// 每个请求的采样配置。开放：新的策略只是在这里增加一个带中性默认值的字段，
/// 而不是新增方法。`temperature == 0.0` ⇒ greedy（退化为 argmax，即当前路径）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32, pub top_k: u32, pub top_p: f32, pub min_p: f32,
    pub repetition_penalty: f32, pub seed: Option<u64>, pub want_logprobs: bool,
}
impl Default for SamplingParams { fn default() -> Self; } // greedy 等价的中性默认值

/// 一次采样调用的结果：每个被采样 row 一个 `SampledToken`（位置由 `ctx.plan().q_lens` 决定）。
#[derive(Debug, Clone, Default)]
pub struct SampleBatch { pub tokens: Vec<SampledToken> }

/// speculative verify 的判定结果。当前是线性链；tree/Medusa 通过 `BatchKind::Spec` 保留，
/// 不放在这里。
#[derive(Debug, Clone)]
pub struct AcceptReject {
    pub accepted_count: Vec<u32>,        // 供 KvEdit 使用（不变量 9）
    pub bonus_token: Vec<SampledToken>,  // 在 rejection 点对 residual（target − draft）采样
}

/// 可插拔、返回概率的 token 选择。它是 STRATEGY trait，不是后端能力——因此它实现于
/// marker 类型（`GreedySampler`、`ChainSampler`）上，所以不在 `Backend` 别名里（不变量 10）。
/// 按构造即对象安全：logits 的 dtype 是模型的具体 `T`（固定在单态化的 `run::<D,T,M>` 入口里），
/// 所以 `Box<dyn Sampler<T, D>>` 是合法的——没有 `DynSampler`/`ErasedLogits`（二者已移除）。
/// 所有 body 都默认由 `MathOps::softmax` 组合出来，因此 greedy 与今天的 argmax 字节级一致，
/// 新后端不需要写任何 sampling 代码。
pub trait Sampler<T: Dtype, D: LlmBackend>: Send + Sync {
    /// 每个序列采样一个 token。scope 随 `ctx` 传入（它在 capture 的 graph 区域之后、在其 stream 上
    /// 入队，这也是 temperature/top-p 能与 graph capture 共存的原因）。
    /// 默认（greedy）：每个 seq 的最后一行 argmax + log_softmax logprob。
    fn sample(&self, logits: &Tensor<T, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<SampleBatch>;
    /// 每个 row 的概率（在温度/top-k/top-p 过滤后），但**不**抽样（spec verify + logprobs）。
    /// 默认：对已缩放 logits 做 softmax。
    fn probs(&self, logits: &Tensor<T, D>, params: &[SamplingParams], out: &mut Tensor<f32, D>, ctx: &StepCtx<'_, D>) -> OpResult<()>;
    /// speculative accept/reject（线性链）。默认：逐位置 ratio test
    /// `r = target_prob[tok]/draft_prob[tok]`，若 `u < min(1,r)` 则接受，否则停止并返回 residual sample。
    /// 对 mask 不敏感：tree 模式从 `ctx.plan().kind` 读取——这个签名不会重塑。
    fn verify(&self, target_logits: &Tensor<T, D>, draft_tokens: &[i32], draft_probs: &Tensor<f32, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<AcceptReject>;
}
```

```rust
// ── application/sampler_stack.rs —— 具体策略（零成本 marker） ──
/// Greedy/argmax：默认路径（每行 temperature==0）。继承所有 trait 默认实现。
pub struct GreedySampler;
impl<T: Dtype, D: LlmBackend> Sampler<T, D> for GreedySampler {}

/// Temperature + top-k + top-p + min-p multinomial。CUDA 可以用融合 kernel 覆写 `sample`；
/// CPU 继承由 softmax 组合出来的默认实现。
pub struct ChainSampler;
impl<T: Dtype, D: LlmBackend> Sampler<T, D> for ChainSampler {
    fn sample(&self, logits: &Tensor<T, D>, params: &[SamplingParams], ctx: &StepCtx<'_, D>) -> OpResult<SampleBatch>;
}
// runtime 持有 `Box<dyn Sampler<T, D>>`（允许的 dyn 接缝 (b)，不变量 11），由参数选出。
```

### 3.7 dtype 与 quant — `domain/dtype/{mod,quant}.rs`, `domain/tensor.rs`

```rust
// ── dtype/mod.rs —— 开放 dtype（不变量 8）。替换 closed `DataType` + numel*SIZE_BYTES dyn 路径。 ──
use half::{bf16, f16};

/// 开放的 dtype 身份（注册表键，不是闭合 match-arm 集合）。内置类型占低位保留 id；
/// 新标量会增量式注册。相等/哈希按 id。约束：`DTypeId` 是固定字节宽度的 STORAGE 标量——
/// 子字节逻辑类型（int4/mxfp4）**绝不是** `DTypeId`，它们是 `QuantScheme.packing`，
/// 通过 `bitcast` 映射到 byte dtype 上。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeId(pub u16);
impl DTypeId {
    pub const F32: DTypeId = DTypeId(0);  pub const F16: DTypeId = DTypeId(1);  pub const BF16: DTypeId = DTypeId(2);
    pub const I32: DTypeId = DTypeId(3);  pub const I8:  DTypeId = DTypeId(4);
    pub const F8E4M3: DTypeId = DTypeId(5); pub const F8E5M2: DTypeId = DTypeId(6);
    pub const U8:  DTypeId = DTypeId(7);  pub const U32: DTypeId = DTypeId(8); // 作为 packed payload 的 bitcast 承载体
    /// 注册一个新的标量（异构后端）；返回保留范围之上的新 id。
    pub fn register(spec: DTypeSpec) -> DTypeId;
    /// 字节宽度—— dyn-typed 路径（loader / binary edge）里的单一真源。
    pub fn size_bytes(self) -> usize;
    pub fn is_float(self) -> bool;
}
#[derive(Clone, Copy, Debug)] pub struct DTypeSpec { pub size_bytes: usize, pub is_float: bool, pub name: &'static str }

/// 编译期 dtype trait（保留主干，不变量 8）：`SIZE_BYTES` + read/write_f64 保持不变，因此
/// `Tensor<T,D>` 的单态化和 `numel*SIZE_BYTES` 计算都不受影响。`DATA_TYPE` 被开放的 `ID` 取代。
pub trait Dtype: Copy + Send + Sync + 'static + std::fmt::Debug {
    const ID: DTypeId;
    const SIZE_BYTES: usize;
    fn read_f64(raw: &Self) -> f64;   // host dequant/debug 的有损扩展
    fn write_f64(v: f64) -> Self;     // host quant/init 的有损收窄
}
```

---

## 6. 保留接缝

这些 ABI 现在已经定型，但故意还没实现。每一项都满足：签名已经容纳它；只是 body /
kernel 分支还在延后。

- **Tree/Medusa mask** —— `BatchKind::Spec { mask: MaskMode::Tree, mask_handle: Option<MaskHandle> }`
  是唯一保留入口（**不要**同时在 `StepCtx` 上再放一个原始指针）。`attention_paged` 已经按
  `plan.kind` 路由；启用 tree 时只需要一个 `MaskMode::Tree` kernel 分支 + 一个非 `None`
  的 `MaskHandle` 生产者。`Sampler::verify` 对 mask 不敏感。延后：tree-attention kernel +
  mask registry。
- **NCCL / gloo collective bodies** —— `CollectiveOps` 的 `Comm` 关联类型和所有方法签名都已
  固定；单 rank 作为恒等实现发货。延后：一个 communicator-bootstrap 函数体（NCCL 初始化
  只在这里）+ 每个方法里的 NCCL 调用。落地时不会改 call site。
- **按 block 的 KV 量化** —— `KvQuantTier::PerBlock { scheme }` 已经在 `PagedKvPool.quant`
  上保留；`KvView` 也已经借用了 `&KvQuantTier`。延后：per-block scale pool + scatter/gather
  的量化分支。不会改 `KvEdit`/`KvView`/attention 的签名。
- **expert-parallel all-to-all** —— `CommAxis::Ep` + `CollectiveOps::all_to_all` 已保留；
  `MoeFfn::run` 就是调用点。延后：dispatch/combine 的 permute。dropless 单 GPU 默认会一直保留。
- **可变形状的 CUDA graph** —— `GraphDecision` 是 enum（不是 bool），因此未来的快速路径
  （captured-Ragged、fused-spec、MoE-with-fixed-capacity）可以作为新 variant 接上。今天只发
  `Graph(slot)` / `Eager`；**MoE decode 只有在 experts 无丢弃且每 expert 容量固定（shape-stable）
  时才可 graph-capture**——这是 `MoeFfn` 的显式要求，不是隐式增量。延后：variable-shape
  capture variants。
- **多流 / 跨阶段 overlap** —— `ExecScope::fork` / `record_event` / `wait_event` / `Event`
  被标成保留接缝（在 trait 上注释掉，这样 v1 没有死方法）。PP overlap 以及 copy/compute
  overlap 在实现时再接入。今天发的是单流。
- **超出 Sum / `ReduceOp::{Max,Min,Avg}`、`broadcast`、`barrier`、`ShardSpec::VocabParallel`、
  `KvEdit::rollback` 语法糖** —— 都保留在 surface 上，以便未来的 sampling-side gather、
  vocab-parallel lm_head 和 chained-spec 便利用 match-arm / 一个方法接上，而不是重塑接口。

---

## 7. 迁移顺序

每一步都能编译。对应 8 个 blocker。**(A)** = 纯增量；**(C)** = 会碰到已有 call site。

1. **(A) 开放 dtype** —— 引入 `DTypeId` registry，同时保留 `Dtype` 主干；添加 `Fp8E4m3`/`Fp8E5m2`。
   将闭合的 `DataType`（domain/types.rs）映射到保留 id；保留现有 `impl Dtype` 不动。
   *(Blocker 8)*
2. **(C) 把 `MemoryPort` 并入 `Device`；重绑 `Tensor<T, D: Device>`** —— 修改 `domain/tensor.rs:20`
  （`D: MemoryPort` → `D: Device`）以及 device 实现。机械改动；会碰到所有 tensor-bound 签名，
   但不改逻辑。*(Blocker 5 foundation, inv 2)*
3. **(C) `ExecScope` 接缝** —— 引入 `<D as Device>::Scope`，把 `CudaConfig` 内部移到 `CudaScope`
   后面，用显式 `scope` 参数替换 `tensor.device().config.stream` 和 diffusion thread-local；
   `enter()` 负责 `cudaSetDevice`。*(Blockers 5, 7)*
4. **(C) 去参数化 `BatchPlan`** —— `domain/batch.rs:34`（`BatchPlan<D: MemoryPort>` → `BatchPlan`）；
   把 device index tensors 移到 `KvIndexTensors`；引入 `StepCtx`。更新 fused-op 调用点去接收 `ctx`。
   *(Blocker 6)*
5. **(C) 拆分 op ports** —— `MathOps`（底座）+ `FusedOps`/`DiffusionOps`（默认组合的上层）；
   把 diffusion 的 `sdpa`/`silu` 合并进 `MathOps`；加上 `#[must_override]` 标记。把
   `attention_paged`/`scatter_kv_paged` 重新表述到 `KvView`/`LayerKv` 上。*(Blocker 6)*
6. **(A) `QuantScheme`** —— 用它替换 `matmul_quant` 里的裸 `group_size`；增加 `bitcast`；
   让 `QuantLinear` 携带 scheme；packed int4 存成 `u32`。*(Blocker 8)*
7. **(C) `Hidden` + `Component`/`DecoderModel`** —— 把 `forward_workspace.rs` 的 god-struct 拆掉，
   变成由 `Runtime` 一次性分配的 `Hidden` 槽位；把 llama3.rs:213-223 的跨层 norm 拆成每个 block
   自己的 norm + `fused_add_rmsnorm`；引入 `embed`/`decode_layers(range)`/`finalize`，并把 `forward`
   作为默认提供。*(Blocker 4)*
8. **(A) `KvEdit` commit** —— pool 持有每个序列的 `kv_len`；把 `seq.kv_len += 1`（decode_engine.rs:264）
   替换为由 `StepOutput.accepted` 驱动的 `KvEdit::apply_step`。*(Blocker 1)*
9. **(A) `Sampler` 策略** —— 把 argmax 从捕获的 graph 中移到 `Box<dyn Sampler<T, D>>`；
   （Greedy 默认 = 字节级一致）；增加 `probs`/`verify`。*(Blocker 2)*
10. **(A) `CollectiveOps` 侧车** —— 添加 trait + 单 rank 恒等实现 + scope 上的
    `TopologyShape`/`Rank`，从保留的 `LoadModel` 字段里填入。rank 1 下没有行为变化。
    *(Blocker 3)*
11. **(C) 二进制边界宏** —— 用 `dispatch_worker!` 代替 `match model_type`（worker_main.rs:242），
    对已经发货的 `(backend, dtype, arch)` tuple 展开；typed `run::<D,T,M>` 入口负责构建
    `Runtime`/`ModelHost`。*(elegance pillar)*

步骤 1、6、8、9、10 在前置步骤落地后都是纯增量；2–5、7、11 会碰到 call site，但不带功能逻辑。

---

## 8. 对比说明

这条主干把现实中的几种做法合在一起，但选了更适合 Rust 单态化引擎的变体。**底座/上层 op
拆分 + 默认组合的 fused ops** 对应 ggml-backend 的“实现一个小 op 集，剩下走通用实现”，
只是这里是静态单态化而不是 vtable 分发；**`Component`/stage-list + carried `Hidden`** 对应
 candle 的模块组合和 HF 风格的 decoder block；**paged KV + `BatchPlan`/ragged tiles + accepted-count
 commit** 对应 vLLM/SGLang 的 continuous batching 和 speculative decode 机制，只是把 vLLM 的
 `BlockManager` 重新表达成 `KvEdit` mutation；**`CollectiveOps`/`TopologyShape` 接缝 + `ShardSpec`**
 对应 TensorRT-LLM / Megatron 的 TP/PP sharding，但被压缩成可增量添加的 impl；而**二进制边界的
 tuple 宏**则是我们对 TensorRT-LLM build-time engine specialization 的回应，只是不借助其 AOT 编译步。
 这里之所以选“静态分发 + 默认组合的 fused ops”，是因为引擎已经承诺用 `Tensor<T, D>` 单态化来换取 kernel
 性能：一个 `dyn` backend（ggml/llama.cpp 风格）会抹掉 CUDA codegen 正需要的类型信息；而一个完全手写的
 per-backend 栈（TensorRT-LLM 风格）又会放弃“新后端只需 ~15 个可移植 op”的增量性。把唯一的 op 主干保持静态，
 并把两个不可避免的 `dyn`（架构对象、sampler 策略）隔离到二进制边界，能同时拿到 vLLM 的能力覆盖面和
 candle 的组合清晰度，又不把热路径的分发税付进去。
