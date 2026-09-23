# GGUF 文件解析设计

日期：2026-09-22。设计已获确认并完成实现，验证结果见 [GGUF_READER_VALIDATION.md](GGUF_READER_VALIDATION.md)。

## 1. 目标与验收边界

新增 CPU 可独立运行的 GGUF reader：打开文件、读取完整元数据、枚举张量、按名称取得经过边界验证的原始字节视图。能够读取当前下载的语言模型和视觉文件，不申请 GPU 显存，也不展开量化权重。

本阶段不包含反量化、CUDA 算子、模型权重名称映射、tokenizer 构建、视觉推理或 worker 启动接入。解析成功仅表示文件结构和张量范围通过检查，不表示模型已经能够推理，也不证明权重数值正确。

## 2. 放置位置与依赖

放在 `crates/infer-worker/src/infrastructure/io/gguf/`，与现有 `safetensors` 并列：

```text
gguf/
  mod.rs       # GgufReader、公开视图、mmap 所有权
  parser.rs    # 字节游标、头部/元数据/张量目录解析与验证
  metadata.rs  # 元数据值与有类型的数组
  types.rs     # GGML 类型 ID 与块布局表
  error.rs     # 带位置与上下文的错误
  tests.rs     # 小型 fixture、边界与损坏文件测试
```

复用已有 `memmap2` 和 `thiserror`，不引入 GGML/C++ 运行库或 CUDA 依赖，不新增 Cargo feature。本阶段不修改 `WeightLoader` 对 `SafetensorsReader` 的依赖；待真正接入 GGUF 模型加载时，再根据两种 reader 的使用需求设计公共接口。

## 3. 首期格式支持

| 项目 | 行为 |
| --- | --- |
| GGUF 版本 | 支持 little-endian v2、v3；v1、未来版本明确报错 |
| Big-endian | 首期不支持；识别已知大端版本编码并返回明确错误，不误作巨大版本/计数 |
| 输入 | 一个明确的文件路径；按 magic 判断，不依赖文件名后缀 |
| 多文件 | 每个文件独立打开；不自动扫描目录、合并分片或寻找 mmproj |
| 元数据 | 全部 13 种 value type；保留整数宽度和符号，不经 JSON 中转 |
| 数组 | 同类型数组、空数组、字符串数组、有限深度嵌套数组 |
| 元数据键 | 保留未知键；不在文件解析层判断模型架构是否受支持 |
| 张量 | 读取名称、GGML 类型、原始维度、相对偏移、绝对偏移、精确字节数 |
| 数据 | 只返回原始字节；不做字节序转换、转置或反量化 |

分片文件可作为独立容器检查，`split.*` 元数据原样暴露；调用方不能把打开一个分片视为完整模型已加载。语言模型与 mmproj 是两个 reader，关联关系属于后续模型加载层。

张量布局注册表覆盖实现时固定版本的上游 `GGML_QUANT_SIZES` 中的有效类型，包括普通浮点/整数、K-quants、IQ 系列及其他已定义块格式。这里只需要每块元素数与存储字节数，不需要实现对应计算。记录上游提交号和来源，避免跟随 master 静默改变。

未知、已移除或缺少可靠布局定义的类型：`open` 返回 `UnsupportedTensorType { id, tensor, offset }`。不使用相邻张量偏移猜测字节长度，因为其中可能包含 padding。后续增加新类型只扩展布局表与测试。

## 4. 接口草案

```rust
pub struct GgufReader { /* mmap + owned index */ }

impl GgufReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError>;
    pub fn open_with_limits(
        path: impl AsRef<Path>, limits: ParseLimits,
    ) -> Result<Self, GgufError>;

    pub fn header(&self) -> &GgufHeader;
    pub fn metadata(&self) -> &GgufMetadata;
    pub fn tensors(&self) -> &[GgufTensorInfo];
    pub fn contains(&self, name: &str) -> bool;
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo>;
    pub fn read_view(&self, name: &str) -> Result<GgufTensorView<'_>, GgufError>;
}

pub struct GgufTensorView<'a> {
    pub info: &'a GgufTensorInfo,
    pub bytes: &'a [u8],
}

// 所有字段由 reader 构造并验证，公开只读 getter。
pub struct GgufTensorInfo {
    name: String,
    ggml_type: GgmlType,
    dimensions: Vec<u64>,
    relative_offset: u64,
    file_offset: u64,
    byte_len: u64,
}
```

`GgufHeader` 暴露版本、字节序、条目数量、alignment、data_offset。元数据使用有类型的 `GgufValue`，数组使用 `GgufArray` 的有类型向量，避免把 tokenizer 的大型数值数组存成逐元素 tagged enum。提供 `get`、`iter` 以及 `as_u32`、`as_str` 等严格类型访问；不自动截断或强制转换。

元数据与张量按文件顺序保存，另建名称索引；重复名称报错，不能悄悄覆盖。返回视图的生命周期绑定 reader，不提供伪造的 `'static` 引用，也不把压缩块伪装成现有标量 `Tensor<T>`。

GGUF 维度保留磁盘顺序，`dimensions[0]` 是连续变化最快的维度。例如磁盘 `[5120, 248320]` 原样返回；后续模型适配层负责解释为本框架的行列顺序。

## 5. 解析与内存策略

1. 打开只读文件并建立 mmap；用显式 little-endian 读取函数读取字段，不进行未对齐的指针转换。
2. 验证 magic、版本、计数；解析元数据并确定 alignment。
3. 解析完整张量目录；根据块布局计算各张量实际字节数。
4. 计算数据区起点，验证所有范围、对齐及重叠，再构建名称索引。
5. `read_view` 只借用已验证的 mmap 切片，调用时不解析整个文件，也不复制张量。

默认 alignment 为 32；显式值必须是 `UINT32`、非零且为 8 的倍数。用通用整除公式向上对齐，不能以位运算暗含“必须是 2 的幂”；测试包含 alignment=24。

设块元素数为 B、块字节数为 S，普通标量取 B=1：

```text
row_bytes = dimensions[0] / B * S
byte_len  = row_bytes * product(dimensions[1..])
absolute_offset = align_up(directory_end, alignment) + relative_offset
```

要求第一维可被 B 整除；不能仅检查总元素数可整除。维度数支持 1～4，各维必须非零；零维标量和空张量首期明确报 `UnsupportedTensorShape`，不作为损坏数据之外的隐式特例。

解析复杂度为 O(元数据字节数 + 张量数 log 张量数)，排序仅用于验证数据区间。权重数据不会被主动遍历或预读；打开 13 GiB 文件不需要分配同等大小的堆内存。字符串和元数据允许复制，内存规模随头部增长。

内部解析函数接受 `&[u8]` 并返回拥有自身数据的索引，便于不依赖文件系统的测试。只有 reader 封装文件映射。保留文件句柄与绝对偏移，为未来按层 positional read 预留信息，本阶段不接入现有预取流水线。

沿用 mmap 的文件稳定性前提：reader 存活期间文件不得被外部原地修改或截断；保留文件句柄并不能阻止这种变化。该前提写入 API 文档，mmap 创建处是集中审查的 unsafe 边界；纯字节解析器使用 safe Rust。

## 6. 验证与错误策略

所有长度、乘加、对齐运算使用 checked arithmetic；转换到 `usize` 前检查可表示性。验证字段边界后才读取或分配，不根据未验证的计数直接 `with_capacity`。

检查项：

- 截断 header、字符串、数组、tensor info 或权重数据；字符串 UTF-8 与布尔值 0/1 合法性。
- 重复 metadata key / tensor name；key 的 ASCII 和长度限制、tensor name 的长度限制。键名词法不强制 lower_snake_case，以兼容扩展键。
- metadata value type 未知时立即失败；未知 key 可保留，但未知 type 无法可靠跳过。
- alignment 类型和值，维度范围，块整除，偏移对齐，范围溢出、越界和张量间重叠。
- 不要求 tensor info 按数据偏移排序；允许范围之间的 padding 和文件尾部额外字节。
- 不要求出现某个架构、tokenizer 或张量名称；允许合法的无张量元数据文件。

`ParseLimits` 是 reader API 参数，不新增 TOML。建议默认：metadata 条目 100,000、tensor 条目 1,000,000、header 最大 256 MiB、单字符串最大 64 MiB、累计数组元素 16,000,000、嵌套数组深度 8、解析索引分配预算 256 MiB。预算同时约束解码后的容器和字符串分配，不能只检查磁盘字节数；大于预算返回 `LimitExceeded`，用户可以显式调整。

使用 `thiserror` 表达 I/O、格式不符、版本/字节序不支持、截断、类型不支持、无效字段、重复名称、溢出、越界、重叠、预算超限、张量不存在等错误。尽可能携带文件路径、字段绝对偏移和 key/tensor 名。错误消息不转储大数组或权重内容。

## 7. 验证计划与完成标准

1. **独立小 fixture**：固定字节样例验证 v2/v3、默认及自定义 alignment、各元数据类型/数组、多个张量和非顺序偏移；至少一份由固定版本上游 GGUFWriter 生成并提交小文件及生成说明，避免读写双方犯同一种错误。
2. **块布局**：逐类型对照上游尺寸表；覆盖本地模型的全部类型，检验多行字节数及错误的行宽。无需反量化正确性测试。
3. **损坏输入**：对小 fixture 逐字节截断，定向修改计数/长度/类型/维度/offset，覆盖溢出、重叠、重复键、非法 bool/UTF-8、深层数组和资源预算；预期返回错误而非 panic。
4. **零复制与生命周期**：验证视图指向映射内的正确区间；Rust 类型约束禁止视图比 reader 活得更久。
5. **本地真实文件**：显式运行的 ignored CPU 集成测试，文件通过环境变量指定；正常 CI 不下载模型、不访问 `~/models`。与独立参考读取结果比较元数据、每个张量的名称/维度/类型/偏移/字节数，并抽查数据切片。
6. **回归**：格式化、无 CUDA 的定向解析测试和现有 reader 测试；不因本次文件解析改动编译/运行 GPU 测试。

2026-09-22 只读检查得到的本地验收基线：

| 文件 | 版本 | 元数据条目 | 张量数 | 数据区起点 | 架构 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-UD-Q3_K_XL.gguf | 3 | 50 | 866 | 10,996,640 | qwen35 |
| mmproj-F16.gguf | 3 | 35 | 334 | 20,224 | clip |

主模型含 F32 和 13 种量化类型：Q2_K/Q3_K/Q4_K/Q5_K/Q6_K、Q8_0、IQ2_XXS/IQ2_XS/IQ2_S、IQ3_XXS/IQ3_S、IQ4_NL/IQ4_XS。视觉文件含 222 个 F32、112 个 F16 张量。这些数量只用于该下载版本的集成测试，不硬编码进 reader。

完成标准：两份文件均能打开、完整列出元数据和张量、按名称取得精确原始字节范围；损坏输入测试通过；现有 Safetensors 读取路径不受影响。此时还不能通过 RustInfer 启动这份模型。

## 8. 实施顺序与待审阅决策

已按“类型与元数据 → 纯字节解析及验证 → mmap reader → fixture/真实文件验证”实施。首期不新增命令行工具，测试和库 API 足以验收。

确认的边界：独立 reader；v2/v3 小端；单文件；完整元数据与已知张量布局；严格范围检查；不包含反量化及推理接入。

实现补充：可配置数组深度仍有 64 层硬上限，避免调用栈溢出；无张量的元数据文件允许省略未使用的数据区 padding。块布局表固定为 llama.cpp `c550d2f60bde72df19fcef1fef627895095b8ba8` 中的 35 种类型。

## 参考

- [GGUF 官方格式规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)：文件结构、元数据、对齐及偏移约定。
- [llama.cpp GGUF 常量与块尺寸](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)：类型 ID 与存储布局。
- [llama.cpp 参考 reader](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py)：用于实施阶段的交叉验证。
- 当前仓库：`infrastructure/io/safetensors.rs`、`models/loader.rs`、`docs/WEIGHT_LOADING.md`。
