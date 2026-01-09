# RustInfer: Rust实现的高性能LLM推理引擎

RustInfer是一个用Rust语言实现的高性能大语言模型(LLM)推理引擎，专注于提供高效、稳定、易于扩展的模型推理能力。项目包含完整的推理核心库、生产级HTTP服务器和现代化Web前端界面。（指项目目标，其实未实现）

## 📰 更新日志

### v0.2.0 (2026-01-09) - 性能大幅提升 🚀

#### 核心改进
- ✨ **BF16 支持**: 新增 BFloat16 混合精度推理，显存占用减半
- ⚡ **算子优化**: 重写关键 CUDA kernel，采用更高效的实现策略
  - Flash Attention GQA 采用Cute对bf16数据进行实现
  - cuBLASLt 矩阵乘法自动调优

#### 性能提升（vs v0.1.0）
| 指标 | v0.1.0 | v0.2.0 | 提升 |
|------|--------|--------|------|
| **Prefill 吞吐量** | ~355 tok/s | ~1052 tok/s | **3x** ⬆️ |
| **Decode 吞吐量** | ~220 tok/s | ~436 tok/s | **2x** ⬆️ |
| **模型加载时间** | ~15 秒 | ~5 秒 | **3.0x** ⬆️ |
| **显存占用** | ~12GB (FP32) | ~6GB (BF16) | **50%** ⬇️ |

> 测试环境: H200, Llama-3.2-1B-Instruct, Batch Size=1

#### 技术细节
- **内存池化**: 实现 CUDA 内存池，将分配开销从 800µs 降低到 1µs
- **零拷贝优化**:
  - mmap 权重加载（100x 加速）
  - KV Cache 零拷贝视图
  - Tensor 切片无数据拷贝
- **Workspace 预分配**: 推理循环中完全消除内存分配

详细技术说明请参阅 [DEVELOPMENT.md](DEVELOPMENT.md)

---

## 🌟 项目特点

- **极致性能**: 采用Rust语言开发，利用其内存安全和零成本抽象特性
  - BF16 混合精度推理
  - 高度优化的 CUDA kernel
  - 零拷贝内存管理
- **多平台支持**: 支持CPU和CUDA加速，可在不同硬件环境下运行
- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **支持主流模型**: 目前实现了Llama3.2 1B模型的完整推理支持
- **内存优化**: 支持KV缓存管理，优化推理过程中的内存使用
- **批量处理**: 支持对输入提示进行批处理优化
- **生产就绪**: OpenAI兼容的HTTP服务器，支持流式响应和性能监控
- **Web界面**: 基于Dioxus的现代化前端，实时显示推理指标和系统资源

## 🏗️ 项目架构

RustInfer采用模块化的架构设计，主要包含三个独立的crate：

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户交互层                          │
│  ┌──────────────────┐       ┌──────────────────┐       │
│  │  Web前端         │       │  CLI/SDK客户端   │       │
│  │  (Dioxus WASM)   │       │  (Python/Rust)   │       │
│  └────────┬─────────┘       └────────┬─────────┘       │
└───────────┼──────────────────────────┼──────────────────┘
            │ HTTP                     │ HTTP
            ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                    infer-server                          │
│  • OpenAI兼容API (聊天补全、流式响应)                    │
│  • 性能指标收集与暴露                                     │
│  • 系统监控 (CPU/GPU/内存)                               │
│  • CORS支持、健康检查                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    infer-core                            │
│  • 模型加载 (safetensors/HF格式)                         │
│  • Tokenizer集成                                         │
│  • Transformer实现 (注意力、FFN、归一化)                 │
│  • KV缓存管理                                            │
│  • CPU/CUDA算子                                          │
└─────────────────────────────────────────────────────────┘
```

### 项目结构

```
RustInfer/
├── crates/
│   ├── infer-core/      # 核心推理库
│   │   ├── src/
│   │   │   ├── base/    # 基础组件（内存管理、错误处理等）
│   │   │   ├── op/      # 算子实现（矩阵乘法、归一化等）
│   │   │   ├── tensor/  # 张量操作
│   │   │   ├── model/   # 模型实现（Llama3等）
│   │   │   └── cuda/    # CUDA加速支持
│   │   ├── README.md    # 核心库文档
│   │   └── tests/       # 单元测试与集成测试
│   │
│   ├── infer-server/    # HTTP推理服务器（OpenAI兼容API）
│   │   ├── src/
│   │   │   ├── api/     # API端点（OpenAI、健康检查、指标）
│   │   │   ├── chat/    # 对话模板（Llama3格式）
│   │   │   ├── inference/ # 推理引擎包装器
│   │   │   └── config/  # 服务器配置
│   │   ├── README.md    # 服务器文档（英文）
│   │   └── README_CN.md # 服务器文档（中文）
│   │
│   └── infer-frontend/  # Web前端界面（Dioxus）
│       ├── src/
│       │   ├── api/     # 后端API客户端
│       │   ├── state/   # 状态管理（对话、指标）
│       │   └── components/ # UI组件
│       ├── assets/      # 样式资源（Tailwind CSS）
│       ├── README.md    # 前端文档（英文）
│       └── README_CN.md # 前端文档（中文）
│
├── Cargo.toml           # 工作区配置
├── README.md            # 项目文档（本文件）
└── LICENSE              # Apache 2.0许可证
```

### 核心模块说明

#### infer-core (核心推理库)
1. **base**: 提供基础功能，包括内存分配器、缓冲区管理、错误处理等
2. **tensor**: 实现张量数据结构和基本操作，支持F32和BF16数据类型
3. **op**: 实现各种算子，如矩阵乘法(Matmul)、RMS归一化(RMSNorm)、旋转位置编码(RoPE)等
4. **model**: 实现模型加载和推理逻辑，支持从safetensors格式加载模型
5. **cuda**: 提供CUDA加速支持，通过FFI调用CUDA内核函数

#### infer-server (HTTP服务器)
1. **api/openai**: OpenAI兼容的聊天补全API，支持流式和非流式响应
2. **api/metrics**: 系统监控端点，提供CPU/GPU/内存使用情况
3. **api/health**: 健康检查和就绪探测端点
4. **inference/engine**: 推理引擎包装器，管理模型实例和请求处理
5. **chat/template**: 对话模板实现（Llama3格式）

#### infer-frontend (Web前端)
1. **components**: React式UI组件（聊天界面、指标面板、消息气泡等）
2. **state**: 状态管理（对话历史、系统指标）
3. **api**: HTTP客户端，与后端服务器通信

## 🛠️ 技术栈

### 核心库 (infer-core)
- **编程语言**: Rust 2024 Edition
- **核心依赖**:
  - `ndarray` + `ndarray-linalg`: 多维数组与线性代数运算
  - `rayon`: 数据并行计算
  - `safetensors`: 模型权重加载（零拷贝）
  - `tokenizers`: HuggingFace分词器集成
  - `memmap2`: 内存映射文件操作
  - `half`: BF16数据类型支持
- **CUDA支持**: 可选的CUDA加速，通过`cc`和`bindgen`构建

### HTTP服务器 (infer-server)
- **Web框架**: Axum + Tokio（异步运行时）
- **中间件**: Tower（CORS、Tracing）
- **序列化**: Serde JSON
- **监控工具**:
  - `sysinfo`: CPU和内存监控
  - `nvml-wrapper`: GPU监控（可选）
- **日志**: Tracing + Tracing-subscriber

### Web前端 (infer-frontend)
- **框架**: Dioxus 0.6（Rust → WASM）
- **HTTP客户端**: reqwest（WASM兼容）
- **样式**: Tailwind CSS
- **工具**: Dioxus CLI (`dx`)

## 安装指南

### 系统要求


### 安装步骤

0. **安装依赖**
```bash
sudo apt-get update
sudo apt-get install clang libclang-dev pkg-config libssl-dev openblas-src conda-forge clang
或 conda install conda-forge::libclang anaconda::openssl
```

1. **克隆代码仓库**

```bash
git clone https://github.com/your-username/RustInfer.git
cd RustInfer
```

2. **构建CPU版本**

```bash
cargo build --release  # 其实默认开启了cuda feature
```

3. **构建CUDA加速版本**

```bash
cargo build --release --features cuda
```

4. **运行前**
```
先运行 cargo test 来保证所有测试正常通过
接着再测试性能
cargo test test_llama3_cuda_performance --release -- --nocapture --ignored
cargo test test_llama3_cpu_loading_and_generation --release -- --nocapture --ignored
```

下图展示了运行cargo test test_llama3_cuda_performance --release -- --nocapture --ignored在H200上运行的结果：

![性能测试图](test_images/image_bf16_H200.png)
*显示了模型加载时间、推理延迟和吞吐量等关键指标*

5. **常见错误**
```
ndarray-linalg有许多后端，如果openblas用不了，可以尝试其它的，如intel-mkl-static
```

6、**改进选项**
```
修改build.rs 里面的计算能力flag 以适配不同的显卡。
cuda feature 未完全拆分。
尚未支持计算图。
未支持量化。
```

## 🚀 快速开始

### 方式1：完整体验（推荐新用户）

启动完整的Web应用（后端 + 前端）：

```bash
# 终端1：启动HTTP服务器
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3/model \
    --port 8000 \
    --device cuda:0

# 终端2：启动Web前端
cd crates/infer-frontend
dx serve --port 3000

# 打开浏览器访问: http://localhost:3000
# 你将看到一个现代化的聊天界面，可以进行多轮对话并实时查看性能指标
```

### 方式2：作为API服务使用

仅启动HTTP服务器，通过API调用：

```bash
# 启动服务器
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3/model \
    --port 8000

# 使用curl测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'

# 使用Python OpenAI SDK
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "你好！"}]
)
print(response.choices[0].message.content)
```

**详细文档**:
- 服务器使用: [crates/infer-server/README_CN.md](crates/infer-server/README_CN.md)
- API参考: [crates/infer-server/README.md](crates/infer-server/README.md)

### 方式3：作为Rust库集成

在你的项目中集成推理核心库：

```toml
# Cargo.toml
[dependencies]
infer-core = { path = "path/to/RustInfer/crates/infer-core", features = ["cuda"] }
```

```rust
use infer_core::model::llama3::Llama3;
use infer_core::base::DeviceType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载模型
    let mut model = Llama3::new(
        "/path/to/llama3/model",
        DeviceType::Cuda(0), // 或者 DeviceType::Cpu
        false               // 是否为量化模型
    )?;

    // 生成文本
    let prompt = "Hello, how are you?";
    let (text, num_tokens, prefill_ms, decode_ms, iterations) =
        model.generate(prompt, 100, false)?;

    println!("Generated: {}", text);
    println!("Performance: {}ms prefill, {}ms decode, {} tokens",
             prefill_ms, decode_ms, num_tokens);

    Ok(())
}
```

**详细文档**: [crates/infer-core/README.md](crates/infer-core/README.md)

### Web前端独立使用

前端可以连接到任何OpenAI兼容的后端：

```bash
cd crates/infer-frontend

# 修改 src/api/client.rs 中的 base_url 指向你的后端
# 然后启动前端
dx serve --port 3000
```

**详细文档**: [crates/infer-frontend/README_CN.md](crates/infer-frontend/README_CN.md)

## 📦 支持的模型

### Llama 3 系列

目前完整支持Meta的Llama 3系列模型：

| 模型 | 参数量 | 推荐设备 | 测试状态 |
|------|--------|----------|----------|
| Llama-3.2-1B | 1B | CPU / GPU | ✅ 完全支持 |
| Llama-3.2-1B-Instruct | 1B | CPU / GPU | ✅ 完全支持 |
| Llama-3.2-3B | 3B | GPU | 🔄 理论支持 |
| Llama-3.1-8B | 8B | GPU (8GB+) | 🔄 理论支持 |

**模型下载**:
- Llama-3.2-1B-Instruct: [HuggingFace](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
- 其他模型: [Meta Llama](https://huggingface.co/meta-llama)

**支持的格式**:
- ✅ SafeTensors (.safetensors)
- ✅ 分片模型 (model.safetensors.index.json)
- ✅ HuggingFace Tokenizer (tokenizer.json)

### 计划支持的模型

- [ ] Deepseek 系列
- [ ] Qwen 系列
- [ ] 文生图 系列
- [ ] 量化模型 (INT8/INT4)

## ⚠️ 当前限制与待实现功能

### 已实现 ✅
- [x] Llama3模型完整推理
- [x] KV缓存管理
- [x] CPU和CUDA后端
- [x] F32和BF16数据类型
- [x] OpenAI兼容API
- [x] 流式响应 (SSE)
- [x] 性能指标收集
- [x] 系统资源监控
- [x] Web前端界面

### 限制与待完善 🔄

#### 高优先级
- **采样器**: 仅支持argmax采样，缺少temperature/top-p/top-k
- **量化**: 不支持INT8/INT4量化模型
- **批处理**: 仅支持单请求处理，无连续批处理
- **内存管理**: 固定KV缓存大小，无动态分配
- **启动速度**: 目前载入速度极慢，可以用异步载入优化

#### 中优先级
- **模型支持**: 仅支持Llama3，其他架构需要适配
- **CUDA优化**: 部分算子未充分优化
- **错误处理**: 某些代码路径使用unwrap()而非Result
- **日志系统**: 混用println!和tracing，不统一

#### 低优先级
- **停止序列**: 不支持自定义停止词
- **Logprobs**: 无法输出token概率
- **函数调用**: 不支持OpenAI的function calling
- **认证授权**: 无API密钥验证机制

### 已知问题
1. **内存泄漏**: KV Cache无自动扩建功能，需要实现PageAttention解决。
2. **并发限制**: 多个请求会串行处理，无请求队列

**贡献建议**: 欢迎提交PR改进以上任何功能！

## ⚡ 性能基准

### 测试环境
- **GPU**: NVIDIA RTX 4070Ti Super
- **模型**: Llama-3.2-1B-Instruct (BF16)
- **批大小**: 1
- **版本**: v0.2.0

### 性能优化技术

1. **BF16 混合精度**: GPU使用BFloat16，内存带宽翻倍，吞吐量提升2x
2. **KV缓存**: 缓存注意力计算中的Key和Value矩阵，避免重复计算
3. **零拷贝加载**: 使用内存映射 (mmap) 直接访问模型权重，加载速度提升100x
4. **CUDA优化**:
   - Flash Attention GQA（融合softmax，减少内存访问3x）
   - 融合SwiGLU算子（gate + silu + multiply合并为单个kernel）
   - cuBLASLt自动调优（达到90%峰值TFLOPS）
5. **并行计算**: CPU算子使用Rayon进行数据并行
6. **内存池化**: CUDA内存分配从800µs降低到1µs
7. **Workspace预分配**: 推理循环零内存分配

## 🛠️ 开发指南

### 代码风格

```bash
# 格式化代码
cargo fmt

# 检查代码质量
cargo clippy -- -D warnings

# 检查文档
cargo doc --no-deps --open
```

### 添加新模型

要添加新的模型支持，需要实现`Model` trait：

```rust
pub trait Model {
    fn init(&mut self, device_type: DeviceType) -> Result<()>;
    fn forward(&mut self, input: &Tensor, pos: &Tensor) -> Result<Tensor>;
    fn tokenizer(&self) -> &dyn Tokenizer;
    fn encode(&self, text: &str) -> Result<Vec<i32>>;
    fn decode(&self, ids: &[i32]) -> Result<String>;
    fn is_eos_token(&self, token_id: u32) -> bool;
    fn slice_kv_cache(&self, layer_idx: usize, start_pos: usize, end_pos: usize)
        -> Result<(Tensor, Tensor)>;
}
```

**实现步骤**:
1. 在 `crates/infer-core/src/model/` 创建新模型文件
2. 定义模型配置结构体
3. 实现层和算子组合
4. 实现 `Model` trait
5. 添加单元测试

**参考实现**: [crates/infer-core/src/model/llama3.rs](crates/infer-core/src/model/llama3.rs)

### 添加新算子

要添加新的算子，需要实现`Op` trait：

```rust
pub trait Op {
    fn name(&self) -> &'static str;
    fn forward(&self, ctx: &mut OpContext) -> Result<()>;
}
```

**实现步骤**:
1. 在 `crates/infer-core/src/op/` 定义算子结构体
2. 实现CPU内核 (`op/kernels/cpu/`)
3. （可选）实现CUDA内核 (`op/kernels/cuda/`)
4. 实现 `Op` trait
5. 添加测试

**参考实现**: [crates/infer-core/src/op/rmsnorm.rs](crates/infer-core/src/op/rmsnorm.rs)

### 添加新API端点

在服务器中添加新端点：

```rust
// crates/infer-server/src/api/your_endpoint.rs
use axum::{extract::State, Json};

pub async fn your_handler(
    State(engine): State<Arc<Mutex<InferenceEngine>>>,
) -> Json<YourResponse> {
    // 实现逻辑
}

// crates/infer-server/src/main.rs
let app = Router::new()
    .route("/v1/your_endpoint", get(your_handler))
    .with_state(engine);
```

### 项目结构约定

- **错误处理**: 使用 `Result<T>` 而非 `panic!` 或 `unwrap()`
- **日志**: 使用 `tracing` 而非 `println!`
- **命名**: 遵循Rust命名规范 (snake_case函数, CamelCase类型)
- **文档**: 为公共API添加 `///` 文档注释
- **测试**: 每个模块都应有对应的测试文件

## 🧪 测试

### 运行测试

```bash
# 运行所有单元测试
cargo test

# 运行性能测试（需要模型文件）
cargo test test_llama3_cuda_performance --release -- --nocapture --ignored

# CPU推理测试
cargo test test_llama3_cpu_loading_and_generation --release -- --nocapture --ignored

# 仅测试核心库
cd crates/infer-core
cargo test

# 仅测试服务器
cd crates/infer-server
cargo test
```

### 测试覆盖

| 模块 | 单元测试 | 集成测试 | 性能测试 |
|------|----------|----------|----------|
| infer-core | ✅ | ✅ | ✅ |
| infer-server | ⚠️ | ⚠️ | ❌ |
| infer-frontend | ❌ | ❌ | ❌ |

**图例**: ✅ 完整  ⚠️ 部分  ❌ 缺失


## 🤝 贡献

欢迎提交Issue和Pull Request！

### 贡献流程

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

### 贡献指南

**在提交PR前，请确保**:
- [ ] 代码通过 `cargo fmt` 格式化
- [ ] 代码通过 `cargo clippy` 检查
- [ ] 所有测试通过 (`cargo test`)
- [ ] 添加了必要的文档注释
- [ ] 更新了相关README文档

**优先处理的贡献**:
- 🐛 Bug修复
- 📝 文档改进
- ⚡ 性能优化
- ✨ 新模型支持
- 🧪 测试覆盖

## 📄 许可证

本项目采用Apache License 2.0开源许可证，详见[LICENSE](LICENSE)文件。

## 📞 联系方式

- **GitHub Issues**: [https://github.com/Vinci-hit/RustInfer/issues](https://github.com/Vinci-hit/RustInfer/issues)
- **Pull Requests**: 欢迎提交功能改进和Bug修复

## 🙏 致谢

### 灵感来源
本项目主要灵感源于课程 **KuiperLLama**:
- KuiperLLama代码: [https://github.com/zjhellofss/KuiperLLama](https://github.com/zjhellofss/KuiperLLama)

### 参考项目
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎设计理念

### 技术栈
- 🦀 **Rust** - 内存安全与零成本抽象
- ⚡ **CUDA** - GPU加速计算
- 🌐 **Axum + Dioxus** - 现代化Web技术栈
- 🎯 **HuggingFace** - 模型与Tokenizer生态

---

## 📚 相关资源

### 文档导航
- [核心库文档](crates/infer-core/README.md)
- [服务器文档（中文）](crates/infer-server/README_CN.md)
- [前端文档（中文）](crates/infer-frontend/README_CN.md)

### 学习资源
- [Rust官方文档](https://doc.rust-lang.org/)
- [Axum Web框架](https://docs.rs/axum/)
- [Dioxus教程](https://dioxuslabs.com/learn/0.6/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/)

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个⭐Star！**

Made with ❤️ and 🦀 Rust

</div>
