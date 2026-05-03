# RustInfer 重构指导：走向 Rusty 的模型层

> 目标：让"添加一个新 LLM 模型"的 diff 只包含**该模型独有的部分**；
> 所有通用的驱动循环、权重加载、状态管理都由抽象层承担。

---

## 一、本文档的用法

本文档分两部分：

- **第 2 节**：Rust 社区公认的 28 条设计原则（作为评审每次改动时的对照表）。
- **第 3 节以后**：结合本项目 `crates/infer-worker/src/model/` 现状，识别具体痛点、给出分阶段重构方案与验收标准。

每次 PR 都应该能回答：

1. 这次改动让哪几条原则得到更好的体现？
2. 这次改动让"再加一个新模型"的成本降低了多少行？

---

## 二、Rusty 的 28 条原则

### A. 类型系统层

1. **Make illegal states unrepresentable**：用 `enum` 表示互斥状态，用字段类型表达约束；不让"理论上可能但不该出现"的组合态存在。
2. **Parse, don't validate**：边界处一次性解析成强类型，之后代码不再重复 `assert!` / 运行时校验。
3. **Newtype pattern**：`usize` / `i32` 到处传会造成语义混淆；用 `LayerIdx(usize)`、`SeqLen(usize)`、`DeviceId(i32)` 让编译器替你 type-check。
4. **Zero-cost abstraction**：抽象不应带来运行时开销。泛型单态化 + `#[inline]` 优于 `Box<dyn>` 于热路径。

### B. 所有权与借用层

5. **Ownership means clarity of responsibility**：谁拥有资源、谁负责释放，签名一眼看出（`&T` / `&mut T` / `T` / `Arc<T>` / `Box<T>` 都是明确合同）。
6. **Prefer borrowing over cloning**：`.clone()` 是兜底，不是设计选择。优先借用、拆分借用、split mutation。
7. **RAII everywhere**：资源（文件、锁、CUDA stream、KV cache 槽位）一律走 `Drop`；不写"请记得调用 release"这种 API。
8. **`&mut` 独占即互斥**：不要用 `RefCell` / `Mutex` 绕过借用检查"让代码编译通过"。那通常意味着模块边界划错了。

### C. 错误处理层

9. **Return `Result`, don't panic**：库代码除真正的内部逻辑 bug（用 `unreachable!` / `debug_assert!`），一律 `Result`。
10. **`?` over `match`**：串联 `Result`/`Option` 用 `?` 链式传播。
11. **One error type per crate/module**：`thiserror` 定义语义枚举，对外只暴露一个 `Error`，用 `From` 做自动转换。

### D. Trait 与组合层

12. **Composition over inheritance**：Rust 没有继承，trait 是 **capability** 而非 is-a。小而正交的 trait 胜过大而全的 trait。
13. **Trait 提供默认实现承载"通用逻辑"**：必要项（required items）要少而正交；通用逻辑挂在默认方法上。
14. **Pure traits for behavior, structs for data**：trait 不携带数据。想在 trait 里"存字段"的冲动，意味着这段逻辑属于某个 struct。
15. **Prefer concrete types and generics over `dyn Trait` on hot paths**：静态分派是零成本的默认选择；`dyn` 只用于真正需要运行时多态处。
16. **Object safety 要显式设计**：一个 trait 如果既要泛型也要 `dyn`，提前考虑 `Self: Sized` 标注与方法兼容性。

### E. API 设计层

17. **Builder pattern for complex constructors**：参数多、多数可选时，用 builder；避免 10 参数的 `new()`。
18. **Accept `impl Trait` / generic bounds, return concrete types**：入参宽松（`impl AsRef<Path>`），返回具体类型。
19. **`From` / `Into` / `TryFrom` / `TryInto` 用于类型转换**：自定义构造命名 `new` / `from_xxx` / `with_xxx`；组合构造链走 `From`。
20. **Make the common case easy, the rare case possible**：最常见调用路径不应需要理解泛型、生命周期、feature flag。

### F. 模块与包组织层

21. **"Common → specific" 倒过来看代码**：公共抽象在父模块，具体实现在子模块；子模块只依赖父模块暴露的抽象。
22. **一个文件只做一件事**：超过 ~500 行且身兼数职，就该拆。
23. **`pub` 尽量窄**：`pub(crate)` 优于 `pub`；`pub(super)` 优于 `pub(crate)`。只把稳定契约对外暴露。
24. **Re-export 定义公共门面**：模块根 `mod.rs` 用 `pub use` 精选对外 API。

### G. 工程实践层

25. **Clippy pedantic + `#![warn(missing_docs)]` 作为基线**：CI 开 `-D warnings`，警告即错误。
26. **测试与实现分离**：`#[cfg(test)] mod tests` 只测公共 API；长时测试 `#[ignore]` + criterion。
27. **Feature flags = 可选能力，不是语法开关**：`#[cfg(feature = "cuda")]` 只应包裹"仅该 feature 下存在"的代码路径。
28. **Document invariants at the type / function level**：`// SAFETY:` / `// Invariant:` 写在代码旁。

---

## 三、当前 `model/llm/` 痛点诊断

结合 `llama3.rs`（862 行）和 `qwen3.rs`（1061 行），按违反的原则分类：

### 痛点 1：`generate` 驱动循环被复制粘贴到每个模型 —— 违反 13、22

`Llama3::generate` 和 `Qwen3::generate` 95% 逻辑相同：

- 创建临时 `BatchWorkspace` + `gen_output` Tensor
- prefill：构造 `WorkerBatchMeta` → 调用 `forward` → 读 `gen_output[0]`
- decode 循环：逐步构造单-token meta → 调用 `forward` → 判 EOS → append → 增量打印
- 最终 `tokenizer.decode`

差异只有：

- Llama3 创建了 `CudaConfig` 并传给 `forward`；Qwen3 传 `None`（因为 Qwen3 自己管理 CUDA graph）
- Llama3 的 workspace 按 `prompt_len` 分配；Qwen3 的 workspace"只为满足 trait 接口"

Qwen3 注释里白纸黑字写着"这里的 BatchWorkspace 仅用于满足统一 trait 接口"——这是**接口 shape 不正确**的直接证据。

### 痛点 2：`impl Llama3` 被迫拆成两块、夹一个 trait impl —— 违反 22

`llama3.rs` 当前顺序：

```
impl Llama3 { new / generate }       // 453 行结束
impl LlmModel for Llama3 { forward }  // 夹心 trait impl
impl Llama3 { compute_worker_batch* } // 第二个 inherent 块
#[cfg(test)] mod tests { ... }
```

这是把 inherent `fn forward` 原地升格为 trait 方法留下的疤痕。**按原则 22，一个文件应先写一整块 `impl Llama3`，再写一整块 `impl LlmModel for Llama3`，再写 `mod tests`**。

### 痛点 3：权重加载 helper 在每个模型里重写一遍 —— 违反 13、21

`load_matmul` / `load_fused_qkv` / `load_fused_gate_up` / `load_rmsnorm` / `load_embedding` / `load_awq_matmul` / `load_fused_gate_up_awq` / `fuse_tensors_vertically`：

- Llama3 和 Qwen3 几乎 1:1 复制（Qwen3 只加了一个统一 `load_weight`）
- 每个方法都不访问 `self`（都是 `fn(loader, device, ...)`）——**说明它们根本不属于模型**
- 正确归属：`impl ModelLoader` 自己提供这些高层装载方法

### 痛点 4：`LlmModel::forward` 签名按 Llama3 shape 回推 —— 违反 12、14

```rust
fn forward(
    &self,
    states: &mut [&mut InferenceState],
    workspace: &mut BatchWorkspace,      // ← Llama3 continuous batching 专属
    batch: &WorkerBatchMeta<'_>,
    output_tokens: &mut Tensor,
    cuda_config: Option<&OpConfig>,
) -> Result<()>;
```

`BatchWorkspace` 是 Llama3 batched forward 的私有中间态，塞到 trait 签名里违反"trait = 行为能力"。Qwen3 确实接受了这个参数但在函数体里只写 `_workspace`。

将来一个 SSM / Mamba / MoE 模型进来，这个签名又要改，又要波及所有调用方。

### 痛点 5：`LoadedModel` 手写 enum dispatch —— 违反 13

`bin/worker_main.rs` 里的 5 个 `match self { LoadedModel::Llama3(m) => ..., LoadedModel::Qwen3(m) => ...}`，每加一个模型要加 5 行。而 `ModelRunner<M: LlmModel>` 本身已经是泛型——外层完全可以：

- 方案 A：`Box<dyn LlmModel>`（单个 vtable 跳转，每步前向一次，开销可忽略）
- 方案 B：`enum_dispatch` crate 宏生成转发

### 痛点 6：测试辅助函数（`warmup` / `generate_and_measure` / `get_dummy_model_path`）在每个模型里重复 —— 违反 26

这些应在 `model/llm/test_utils.rs`（`#[cfg(test)]`）里集中一次。

---

## 四、分阶段重构路线图

### 阶段 0：排版修正（半小时，零风险）

**动机**：消除痛点 2 的夹心 impl。

**动作**：
- 把 `llama3.rs` 中间的 `impl LlmModel for Llama3` 整段挪到文件末尾（`mod tests` 之前）。
- 合并被拆成两块的 `impl Llama3` 为一块。
- `qwen3.rs` 检查同样问题。

**验收**：
- `impl Llama3 { ... }` 只有一个块
- `impl LlmModel for Llama3 { ... }` 紧随其后
- `#[cfg(test)] mod tests { ... }` 在最后
- `cargo check --all-targets --features cuda` 通过

---

### 阶段 1：把 `generate` 提成 trait 默认方法（收益最大）

**动机**：消除痛点 1、部分消除痛点 4。

**前置动作**：先把 `forward` 签名里的 `BatchWorkspace` 去掉——把它移进 `InferenceState`（或一个新 `ForwardContext`）里，作为模型内部状态。让 `forward` 签名瘦身为：

```rust
fn forward(
    &self,
    states: &mut [&mut InferenceState],
    batch: &WorkerBatchMeta<'_>,
    output_tokens: &mut Tensor,
) -> Result<()>;
```

CUDA config 也挪进 `InferenceState`（本来 Qwen3 就这么干了）。

**动作**：

1. 在 `LlmModel` trait 上增加默认方法：

   ```rust
   pub trait LlmModel: Send + Sync {
       // 必须项
       fn config(&self) -> &RuntimeModelConfig;
       fn tokenizer(&self) -> &dyn Tokenizer;
       fn device_type(&self) -> DeviceType;
       fn forward(&self, states: &mut [&mut InferenceState],
                  batch: &WorkerBatchMeta<'_>,
                  output_tokens: &mut Tensor) -> Result<()>;

       // 可选定制（带默认）
       fn create_state(&self) -> Result<InferenceState> { /* 默认 */ }
       fn fill_rope_cache(&self, ...) -> Result<()> { /* 默认 */ }

       // 通用高层 API（全部默认，**不应被模型 override**）
       fn generate(&self, state: &mut InferenceState, prompt: &str,
                   max_tokens: usize, print_output: bool)
           -> Result<GenerateStats>
       { /* 统一实现 */ }
   }
   ```

2. 返回类型用结构体而非元组（原则 1、3）：

   ```rust
   pub struct GenerateStats {
       pub text: String,
       pub num_tokens: u32,
       pub prefill_ms: u64,
       pub decode_ms: u64,
       pub decode_iterations: usize,
   }
   ```

3. 删除 `Llama3::generate` / `Qwen3::generate` 两份实现（共 ~250 行）。

**验收**：
- 两份 `generate` 被删除
- 新模型实现 `LlmModel` 后自动拥有 `model.generate(&mut state, prompt, max, true)`
- 现有所有测试保持通过

---

### 阶段 2：消除 `LoadedModel` 手写 dispatch

**动机**：痛点 5。

**选型**：二选一
- **选 A**：`pub type LoadedModel = Box<dyn LlmModel>;`
  - 优点：不引入宏、不加依赖、添加新模型 = `Box::new(Qwen3::new(...)?)` 一行
  - 代价：每步 forward 一次 vtable dispatch（对 ≥100 tokens/sec 的模型来说完全不可测）
- **选 B**：`enum_dispatch = "0.3"`
  - 优点：静态分派零开销
  - 代价：引入一个 proc-macro 依赖

默认推荐 A（最符合原则 20：common case easy）。

**验收**：
- `bin/worker_main.rs` 里的 `enum LoadedModel` 和 5 个 `match` 全部删除
- 添加新模型不需要修改 `worker_main.rs`

---

### 阶段 3：权重加载 helper 归入 `ModelLoader`

**动机**：痛点 3。

**动作**：

在 `model/common/` 下新建 `weight_loader.rs`（或直接 `impl ModelLoader`）：

```rust
impl ModelLoader {
    pub fn load_tensor_to_device(&self, name: &str, device: DeviceType) -> Result<Tensor>;
    pub fn load_matmul(&self, name: &str, device: DeviceType) -> Result<Matmul>;
    pub fn load_rmsnorm(&self, name: &str, device: DeviceType, eps: f32) -> Result<RMSNorm>;
    pub fn load_embedding(&self, name: &str, device: DeviceType) -> Result<Embedding>;

    /// Fused Q/K/V from three separate tensors at layer `layer_idx`.
    pub fn load_fused_qkv(&self, layer_idx: usize,
                          dims: QkvDims, device: DeviceType) -> Result<Matmul>;

    /// Fused gate_proj + up_proj.
    pub fn load_fused_gate_up(&self, layer_idx: usize,
                              dims: GateUpDims, device: DeviceType) -> Result<Matmul>;

    // AWQ 变体：
    pub fn load_awq_matmul(&self, name_prefix: &str, device: DeviceType,
                           group_size: usize) -> Result<Matmul>;
    pub fn load_fused_gate_up_awq(&self, layer_idx: usize, dims: GateUpDims,
                                  device: DeviceType, group_size: usize) -> Result<Matmul>;
}
```

`QkvDims`、`GateUpDims` 是 newtype（原则 3），避免 `q_dim: usize, kv_dim: usize, dim: usize` 位置错用。

**验收**：
- 两个模型 `new()` 里不再出现 `Self::load_*` 辅助函数调用
- `fuse_tensors_vertically` 这种纯工具函数归入 `ModelLoader` 或独立 `util` 模块

---

### 阶段 4：抽象 `DecoderLayer`（等第三个 LLM 加入时再做）

**动机**：彻底消除 Llama/Qwen 模型结构的重复。

**动作**：

```rust
pub struct DecoderLayer {
    pub rmsnorm_attn: RMSNorm,
    pub wqkv: Matmul,
    pub wo: Matmul,
    pub mha: FlashAttnGQA,
    pub rope: RoPEOp,

    pub rmsnorm_ffn: RMSNorm,
    pub w_gate_up: Matmul,
    pub w2: Matmul,
    pub swiglu: SwiGLU,

    // Qwen3-style QK-norm 可选扩展
    pub q_norm: Option<RMSNorm>,
    pub k_norm: Option<RMSNorm>,
}

impl DecoderLayer {
    pub fn load(layer_idx: usize, loader: &ModelLoader,
                cfg: &RuntimeModelConfig, opts: DecoderLayerOptions,
                device: DeviceType) -> Result<Self>;
    pub fn forward_batch(&self, ctx: &mut LayerForwardCtx) -> Result<()>;
}
```

`Llama3` / `Qwen3` 退化成瘦包装：

```rust
pub struct Llama3 {
    config: RuntimeModelConfig,
    device_type: DeviceType,
    tokenizer: Box<dyn Tokenizer>,
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RMSNorm,
    cls: Matmul,
}
```

**提示**：本阶段**不要预先做**。先等第三个 LLM 进来；抽象必须由 ≥3 个具体用例驱动，否则容易变成"为了设计而设计"的过度工程（违反原则 20）。

---

### 阶段 5：`forward` 签名规范化（与阶段 1 合并落地）

如果阶段 1 已把 workspace/cuda_config 内化进 `InferenceState`，本阶段无需额外动作。否则需要：

- `trait LlmModel::forward` 只接受"外部视角"必要的参数：输入 batch、输出 buffer。
- CUDA/workspace 资源作为 `&mut InferenceState` 的内部细节。
- `InferenceState` 本身可引入 trait `ModelState` 如果未来 SSM 需要不同形状的状态。

---

## 五、验收矩阵：新加一个 LLM 应该长什么样

终态下，加一个 LLM `Foo` 的 diff 应该是：

```rust
// crates/infer-worker/src/model/llm/foo.rs（新文件 ~100-150 行）

pub struct Foo {
    config: RuntimeModelConfig,
    device_type: DeviceType,
    tokenizer: Box<dyn Tokenizer>,
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RMSNorm,
    cls: Matmul,
}

impl Foo {
    pub fn new<P: AsRef<Path>>(model_dir: P, device_type: DeviceType) -> Result<Self> {
        let loader = ModelLoader::load(model_dir.as_ref())?;
        // ... 30 行装载 ...
        Ok(Self { config, device_type, tokenizer, embedding, layers, final_norm, cls })
    }
}

impl LlmModel for Foo {
    fn config(&self) -> &RuntimeModelConfig { &self.config }
    fn tokenizer(&self) -> &dyn Tokenizer { self.tokenizer.as_ref() }
    fn device_type(&self) -> DeviceType { self.device_type }

    fn forward(&self, states, batch, output) -> Result<()> {
        // ~50 行：调用 layer.forward_batch loop + final_norm + cls + sample
    }
}
```

外加 `mod.rs` 里一行 `pub mod foo;`。**不需要改 worker_main.rs、不需要改 scheduler、不需要写 generate/warmup**。

---

## 六、每次 PR 的自检清单

- [ ] 本次改动让 `add-new-model` diff 减少了 N 行（或保持）？
- [ ] 有没有引入新的"跨模块共享的复制粘贴"？
- [ ] 是否违反了 28 条原则中的某一条？写在 PR description 里。
- [ ] 是否 `cargo check -p infer-worker --all-targets --features cuda` 通过？
- [ ] 是否 `cargo clippy -- -D warnings` 通过（目标在阶段 0 完成后开启）？
- [ ] 公开项是否必须 `pub`？能不能 `pub(crate)`？
- [ ] 新加的 trait 方法是必须项还是带默认实现？默认实现是否覆盖了常见场景？

---

## 七、阶段优先级与预期工作量

| 阶段 | 原则对齐 | 状态 | 预期工作量 | 风险 | 依赖 |
|---|---|---|---|---|---|
| 0 - 排版修正 | 22 | ✅ 已完成 | 0.5h | 极低 | - |
| 1 - `generate` 提成 trait 默认方法 | 13, 14 | ✅ 已完成 | 0.5-1d | 中（要调 `forward` 签名） | 阶段 0 |
| 2 - `LoadedModel` 改 `Box<dyn>` | 13, 20 | ✅ 已完成 | 1-2h | 低 | 阶段 1 |
| 3 - 权重加载归入 `ModelLoader` | 13, 21 | ✅ 已完成 | 0.5d | 低 | - |
| 4 - `DecoderLayer` 抽象（Llama3 先行） | 1, 12, 13, 28 | ✅ Llama3 已完成 / ⏳ Qwen3 待做 | 2-3d | 高（大面积改动） | 阶段 3 |
| 5 - `forward` 签名定型 | 12, 14 | ⏳ 留作独立重构 | 2-3d | 中（触及 Runner） | - |

建议先做 **阶段 0 → 阶段 1 → 阶段 2 → 阶段 3**，然后等新模型进来再做阶段 4。
