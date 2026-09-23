# 第一步：统一块量化权重表示

日期：2026-09-22。第一步设计已获确认并完成实现。

2026-09-23 更新：CPU 反量化、Embedding 与 Linear 计算已完成，见 [CPU 阶段实现与验证](BLOCK_QUANT_CPU.md)。下文保留第一步的阶段边界；其中“计算未实现”描述的是第一步完成时的状态。

## 1. 本阶段交付

让框架能够明确表示“某种格式的压缩矩阵”，并让 Linear、Embedding 持有它。压缩数据、逻辑形状、块布局与生命周期都有可验证的约束；同一投影的不同输出段可以保持不同格式。

本阶段包含：13 种格式的描述、主机借用视图、设备存储对象、行/块定位、GGUF 到权重视图的适配、Linear/Embedding 数据结构接入、现有稠密调用点迁移及测试。

本阶段不实现反量化、量化矩阵乘、token gather、完整模型装配、MTP 或多卡量化执行。调用尚未实现的量化前向计算会在写入输出前明确返回 `Unsupported`。完成后应能构建和检查权重对象，而不是生成文本。

## 2. 当前代码带来的约束

- `LinearWeight<T, D>` 已有 Dense、Awq、Fp8Block，`LmHead` 复用 Linear。
- `Embed<T, D>` 当前暴露 `pub table: Tensor<T, D>`。主干构建、共享输出权重、MTP、EAGLE3、DFlash 和部分测试直接访问这个字段，必须一起迁移。
- `AttentionCore.qkv_proj`、`DenseFfn.gate_up_proj` 都是一个 Linear；现有构建器会合并权重。
- GDN 构造器校验稠密权重形状；增加块量化分支时保留稠密路径，为新格式读取逻辑矩阵形状。AWQ/FP8 的 GDN 支持属于独立适配。
- `Tensor<u8, D>` 已可用，底层 `Arc<Storage<D>>` 支持共享。原始字节不需要再新增一套分配器，也不能当作普通 I8 数值矩阵执行。
- `MemoryPort::upload` 的契约允许异步，因此不能仅依赖一个借用参数就认定上传返回后可以释放 mmap。

## 3. 分层与依赖

```text
GGUF reader（文件字节、GGML type ID、磁盘维度）
       │
       ▼
worker/models/gguf_weights.rs（格式映射、维度解释）
       │
       ▼
infer-core（块格式、矩阵布局、主机视图、设备权重）
       │
       ▼
worker/components（Linear / Embedding / 输出段组合）
```

`infer-core` 不知道 GGUF 文件、mmap、Hugging Face 张量名或 Qwen 层号。编码格式按 GGML 兼容的块布局定义，但格式枚举不使用 GGML 文件 ID 作为内部 ABI。

建议文件：

| 位置 | 职责 |
| --- | --- |
| `infer-core/src/dtype/quant/block.rs` | `BlockQuantFormat`、块元素数/字节数 |
| `infer-core/src/quantized.rs` | `BlockQuantLayout`、`BlockQuantView`、`BlockQuantWeight` |
| `infer-worker/src/models/gguf_weights.rs` | GGUF 编码映射、二维矩阵借用视图 |
| `infer-worker/src/components/block_quant_projection.rs` | 一段或多段量化矩阵的输出行组合 |
| 现有 `components/linear.rs`、`embed.rs` | 新权重分支、查询方法、显式未实现错误 |

复用 `OpError/OpResult`、`MemoryPort`、`Tensor`，不新增依赖或配置项。现有 AWQ `QuantScheme` 继续描述它自己的 scales/zeros 布局；GGML 的块内元数据保存在压缩字节中，不拆成 AWQ 的外部 scales/zeros。

## 4. 格式与形状

`BlockQuantFormat` 首期只包含当前模型需要的 13 种。编码语义固定为所采用上游版本的 little-endian 块表示。

| 格式 | 每块逻辑元素 B | 每块存储字节 S |
| --- | ---: | ---: |
| Q2_K | 256 | 84 |
| Q3_K | 256 | 110 |
| Q4_K | 256 | 144 |
| Q5_K | 256 | 176 |
| Q6_K | 256 | 210 |
| Q8_0 | 32 | 34 |
| IQ2_XXS | 256 | 66 |
| IQ2_XS | 256 | 74 |
| IQ2_S | 256 | 82 |
| IQ3_XXS | 256 | 98 |
| IQ3_S | 256 | 110 |
| IQ4_NL | 32 | 18 |
| IQ4_XS | 256 | 136 |

这些常量以现有 reader 固定的 llama.cpp `c550d2f60bde72df19fcef1fef627895095b8ba8` 为依据。实现时让 reader 对这 13 种类型的 `layout()` 委托到 core 常量，避免维护两套块尺寸；reader 其余类型的解析能力仍然保留。

“reader 能识别 35 种存储布局”和“执行侧能表示 13 种量化格式”分别处理：其余类型转换到本阶段块量化对象时返回 `Unsupported`；F32/F16/BF16 走稠密权重路径，不作为量化格式。

所有权重的逻辑形状统一为 `[N, K]`：N 为输出行数，K 为输入列数。Embedding 的 N 是词表大小，K 是隐藏维度。线性层语义仍为 `X[M,K] × W[N,K]^T → Y[M,N]`。

GGUF 的二维磁盘维度 `[K,N]` 在适配层解释为 `[N,K]`，这只更改形状解释，不搬动字节。三维/四维张量不会被这个矩阵入口自动展平，留给对应模型组件明确处理。

## 5. 核心对象与接口草案

```rust
// 字段私有。格式决定块布局；调用方不能另外指定矛盾的块尺寸。
pub struct BlockQuantLayout {
    format: BlockQuantFormat,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    byte_len: usize,
}

impl BlockQuantLayout {
    pub fn new(format: BlockQuantFormat, rows: usize, cols: usize)
        -> OpResult<Self>;
    pub fn shape(&self) -> [usize; 2];
    pub fn format(&self) -> BlockQuantFormat;
    pub fn row_bytes(&self) -> usize;
    pub fn byte_len(&self) -> usize;
    pub fn blocks_per_row(&self) -> usize;
    // 以下范围是相对于当前权重字节视图的半开区间。
    pub fn row_byte_range(&self, rows: Range<usize>)
        -> OpResult<Range<usize>>;
    pub fn block_byte_range(&self, row: usize, blocks: Range<usize>)
        -> OpResult<Range<usize>>;
}

pub struct BlockQuantView<'a> {
    layout: BlockQuantLayout,
    bytes: &'a [u8],
}

pub struct BlockQuantWeight<D: MemoryPort> {
    layout: BlockQuantLayout,
    bytes: Tensor<u8, D>, // 物理形状 [byte_len]，连续
}
```

`BlockQuantView::new(layout, bytes)` 校验切片长度，`slice_rows(range)` 返回借用子视图。文件适配函数返回这个对象，其生命周期受 `GgufReader` 约束。

`BlockQuantWeight::try_new(layout, bytes)` 校验设备 Tensor 的物理形状和连续性。`from_host(view, device)` 显式分配和复制；`slice_rows(range)` 返回共享已有 Storage 的设备子视图；`Clone` 只共享 Storage，不复制数据。提供只读的 `layout()`、`bytes()`、`device()`，不提供修改格式或形状的 setter。

约束与计算：

```text
N > 0, K > 0
K % B == 0
row_bytes = (K / B) * S
byte_len  = N * row_bytes
physical_bytes.len() == byte_len
```

N×K、行字节数、总字节数及所有区间运算均检查溢出。GGUF u64 维度先执行受检的 usize 转换。构造失败返回错误，不触发底层 Tensor 的断言。

行切片只允许完整行；不提供任意列切片、转置或量化块内的切片。`row_byte_range`/`block_byte_range` 可返回合法空区间，但 `slice_rows` 构造新矩阵时拒绝空行集合，维持 N>0 的不变量。子视图的行号从零开始，字节范围相对子视图；不得把底层 Storage 的偏移重复累加。

字节对象只保证编码长度/布局有效，不保证量化块中浮点 scale 的数值合理。具体数值验证属于下一步反量化测试。未来 kernel 也不能假定每块或每行都按 16/32 字节对齐，例如 Q3_K 的块长为 110 字节。

## 6. 所有权与上传

主机视图可以零复制地借用 mmap；设备对象拥有 Arc 管理的分配，两种对象不混用“存储位置”布尔标志。

本阶段 `from_host` 采用完成后返回的语义：上传后显式等待设备复制完成，期间保持输入视图有效。错误路径同样需要完成/收束已提交的复制，才能允许源借用结束；不得在可能仍访问源指针时返回。暂不提供异步上传 API。这样成功返回后即可释放 reader，设备权重保持有效。

实际设备 ID 需要参与投影组合和执行上下文检查；仅有 `D=Cuda` 并不代表两个权重位于同一张卡。CPU 测试验证复制与共享存储，实际 CUDA 上传验证安排在接入 GPU 阶段。本阶段不把整个 12 GiB 模型上传到设备。

## 7. Linear：一段与多段使用同一新分支

```rust
pub struct BlockQuantProjection<D: MemoryPort> {
    // 非递归；每段只有自己的单一格式和 [N_i,K]。
    parts: Vec<BlockQuantWeight<D>>,
    // 构造时计算的逻辑输出行范围。
    output_rows: Vec<Range<usize>>,
    rows: usize,
    cols: usize,
}

pub enum LinearWeight<T: Dtype, D: LlmBackend> {
    Dense(Tensor<T, D>),
    Awq { /* 保留现有定义 */ },
    Fp8Block { /* 保留现有定义 */ },
    BlockQuant(BlockQuantProjection<D>),
}
```

`BlockQuantProjection::try_new(parts)` 要求非空、各段 K 相同、位于同一设备，使用受检加法计算 N 总和与输出范围。一段即普通投影，多段即按输出行拼接；不存在嵌套组合或一个元素一个格式的情况。本阶段组合段仅支持块量化权重；稠密/AWQ/FP8 混合段不在当前模型需求中。

例：实际第 3 层全注意力权重为：

| 段 | 格式 | 逻辑形状 | 输出列范围 |
| --- | --- | --- | --- |
| Q 投影（含模型的 gate 输出） | IQ4_NL | [12288,5120] | [0,12288) |
| K | Q4_K | [1024,5120] | [12288,13312) |
| V | Q5_K | [1024,5120] | [13312,14336) |

只构建三个存储引用和组合元数据，不拼接或重新量化底层字节。组合顺序由调用者显式给定；组件不猜测 Q/K/V 语义，也不改写 Q/gate 的内部排列。gate/up 同样可用两段表示。

保留 `AttentionCore` 与 `DenseFfn` 的 Linear 字段类型。增加量化构造入口；`out_features()` 返回总 N，补充统一的逻辑形状查询；`as_dense()` 对新分支返回 None。bias 仍是激活类型 T 的向量，要求长度为总 N、同设备。

本阶段 `Linear::forward` 遇到新分支先校验形状、设备和 TP1，再返回 `Unsupported`，保证输出不变。后续执行接口以单个 `BlockQuantWeight` 为计算单位，由 Linear 按组合顺序分发。

多 token 时，输出段通常是 stride=[总N,1] 的视图，不是紧密排列的 `[M,N_i]` 缓冲。阶段三必须选择支持该 stride 的 kernel，或用连续临时输出再拷贝；不能将它假装成连续 Tensor。本阶段记录段范围并测试 M>1 的地址关系，不实现计算或 scratch 分配。

## 8. Embedding 与旧调用点迁移

```rust
pub enum EmbeddingWeight<T: Dtype, D: LlmBackend> {
    Dense(Tensor<T, D>),
    BlockQuant(BlockQuantWeight<D>),
}

pub struct Embed<T: Dtype, D: LlmBackend> {
    weight: EmbeddingWeight<T, D>,
    parallelism: EmbeddingParallelism,
}
```

保留 `Embed::new(dense_table)`，新增 `from_block_quant(weight)`。对外提供 `weight()`、`shape()`、`device()`、`as_dense()` 和返回显式错误的 `require_dense()`；提供共享权重的 shallow clone 操作。量化 Embedding 首期只接受单段矩阵，以固定 row_bytes 定位 token。

迁移规则：

| 现有用法 | 新用法 |
| --- | --- |
| `.table.shape()` / `.table.device()` | `Embed::shape()` / `device()` |
| 稠密专用的草稿模型构建、测试数据指针访问 | `require_dense()`，量化输入显式返回 Unsupported |
| 克隆 Embedding 给 MTP 等组件 | shallow clone，共享存储与并行元数据 |
| 共享 Embedding 作为输出层 | 按枚举匹配：Dense 共享 Tensor；BlockQuant 包成单段投影共享 Storage |

现有 Safetensors 构建器调用旧的 dense constructor，既有 TP2 稠密行为保持不变。量化 Embedding/Linear 本阶段明确限制为 TP1；包括调用 `with_parallelism` 后改变配置的情况，forward 仍须检查。不能默默把量化 Embedding 解码成整张稠密词表。

本模型的 Embedding 与输出层实际是不同张量、不同格式；只有模型明确声明共享且映射正确时才共享，不能根据形状相同自动绑定。

## 9. GGUF 适配入口与后端边界

建议入口：

```rust
pub fn block_quant_view<'a>(view: GgufTensorView<'a>)
    -> OpResult<BlockQuantView<'a>>;
```

入口只做类型转换、二维形状解释及长度校验。不会打开文件、猜层号、加减 Norm 的 1、转换 A_log、合并投影或自动上传。文件名、偏移及张量名用于适配错误上下文，不放进核心权重对象。

这一阶段不增加 `MathOps` / `CoreOps` 的空计算方法。阶段二使用 `BlockQuantView` 建立 CPU 解码参考；阶段三再以 `BlockQuantWeight` 为参数增加后端计算能力，届时明确输入 dtype、累加精度、输出 stride 与 scratch 契约。

因此 `BlockQuantFormat` 当前只表示编码能力，不表示某个 CPU/CUDA 后端已能执行它；不得用 `from_id` 或构造成功作为 kernel 可用性的判断。

## 10. 验收与实施顺序

1. **格式与布局**：13 种类型与固定的上游 fixture 对照；检查第一维对应的块整除、逻辑/物理形状、尺寸溢出。
2. **行与块视图**：首尾行、单行、非零起点、越界、空区间、子视图再次切片；字节位置与参考清单一致。
3. **所有权**：主机视图不能超出 reader 生命周期；CPU 上传完成后释放源仍可读取；设备权重 clone/切片共享 Storage，释放原对象不使子视图失效。
4. **组合投影**：使用真实第 3 层 Q/K/V 的形状与三种格式验证输出范围；拒绝空段集、不同 K、跨设备和总 N 溢出。设备检查用可区分设备 ID 的测试后端验证，无需 GPU。
5. **组件接入**：Linear/Embedding 新分支保留权重格式；执行返回 Unsupported 且输出哨兵未变化；量化 TP2 拒绝，旧 dense 路径仍通过。
6. **真实文件检查**：显式 ignored 测试枚举已下载主文件的 506 个量化张量并创建借用矩阵描述，不复制整个模型；抽取少量完整行做 CPU 存储对象验证。F32 及视觉 F16 张量不强行转为块量化对象。
7. **回归**：GGUF fixture、原 Linear/Embedding/GDN、共享 Embedding/输出层、MTP 及草稿组件相关 CPU 测试；默认 CUDA feature 下 `cargo check` 确认 dense-only 调用点迁移完整，不运行无关 GPU 性能测试。

实施按“core 格式/布局 → 主机/设备权重 → GGUF 适配 → 投影组合 → Linear/Embedding 及调用点迁移”分块推进。

本次审阅的核心决策：**逻辑二维矩阵 + 一维压缩字节；主机借用与设备所有权分离；同一 Linear 通过多段量化矩阵保留混合格式；Embedding 改为显式权重枚举；计算接口留到对应阶段实现。**

## 11. 实现细节与验证记录

- `BlockQuantView` 只有布局和借用切片，无设备管理或上传方法。设备上传入口在 `BlockQuantWeight::from_host`。
- `from_host` 使用一个有所有权的临时 host buffer，再上传、同步。这样即使后端同步失败，也不会把仍可能被 DMA 访问的 mmap 借用交还调用者。同步失败返回 `OpError::Fatal` 并保留两端缓冲，等待进程退出回收；上传失败但同步成功则正常释放。这是错误安全所需的显式 staging 副本，不是完整模型的持久稠密副本。
- 通过已有设备分配接入的 `try_new` 不复制；`Clone` 和行切片共享 Storage。
- `Embed::shared_linear` 只提供显式 replicated TP1 权重共享。原有 Safetensors 的 TP2 共享路径通过 `require_dense()` 取得原 Tensor 后继续调用原词表分片构建器。
- CUDA 编译检查发现先前 INT4 测试缺少输出 dtype 标注，已补充 `Tensor<bf16,Cuda>`；本阶段未运行该量化计算测试。

验证结果：

| 检查 | 结果 |
| --- | --- |
| Core 布局/区间测试 | 2 项通过 |
| Worker 单元测试 | 178 项通过，含 GDN、共享权重、MTP 与原 reader 回归 |
| 新权重集成测试 | 8 项通过，含延迟复制、错误路径、跨设备拒绝和混合投影 |
| GGUF 上游 fixture | 1 项通过，35 种布局仍一致 |
| DFlash / EAGLE3 / 混合解码器 | 4 / 10 / 23 项通过 |
| 真实主模型 | 506 个量化矩阵全部创建借用描述，并各复制一整行到 CPU 校验 |
| 生命周期 compile-fail doctest | 通过 |
| 默认 CUDA feature 下 Worker bins/tests | `cargo check` 通过 |

上述共 228 次测试通过（包含显式运行的真实文件测试与 doctest）。正常测试不会自动访问大模型。复现：

```bash
cargo test -p infer-core --lib quantized
cargo test -p infer-worker --no-default-features --lib
cargo test -p infer-worker --no-default-features --test block_quant_weights --test gguf --test dflash --test eagle3 --test hybrid_decoder
cargo test -p infer-worker --no-default-features --doc block_quant_view
RUSTINFER_GGUF_MODEL=/absolute/path/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  cargo test -p infer-worker --no-default-features --test block_quant_weights block_quant_local_model -- --ignored --nocapture
cargo check -p infer-worker --bins --tests
```

CUDA 检查需要本机 CUDA 编译环境。尚无 GPU 权重上传或量化前向运行结论；下一阶段在这些布局和视图上实现参考反量化。
