> 🚧 **radix_tree 分支重构中**: 正在实现 RadixTree + 分布式架构的新技术路线，详见 `radix_tree` 分支。

# RustInfer: Rust实现的高性能LLM推理引擎

RustInfer是一个用Rust语言实现的高性能大语言模型(LLM)推理引擎，专注于提供高效、稳定、易于扩展的模型推理能力。

> **测试环境**: H200, Llama-3.2-1B-Instruct, Batch Size=1

---

## ✨ 核心特性

- **极致性能优化**
  - 内存池化：CUDA分配开销从 800µs → 1µs（800x加速）
  - 零拷贝模型加载：mmap直接映射（100x加速）
  - Workspace预分配：推理循环零内存分配
  - CUDA Graph捕获：kernel启动开销降低10-100x

- **先进的推理技术**
  - BF16混合精度推理（2x带宽，2x吞吐量）
  - Flash Attention GQA（减少3x内存访问）
  - 融合算子（SwiGLU, RoPE等）
  - cuBLASLt自动调优（90%峰值TFLOPS）
  - KV缓存零拷贝视图

- **生产级架构**
  - 进程分离设计（引擎与服务器解耦）
  - ZeroMQ IPC通信（低延迟，高可靠）
  - OpenAI兼容API
  - 流式响应支持（SSE）
  - 实时性能监控

- **现代化技术栈**
  - Rust 2024内存安全保证
  - 模块化Crate设计
  - CPU和CUDA双后端
  - Web前端界面（Dioxus WASM）

📖 **开发者文档**: 想深入了解设计哲学、架构细节和实现原理？请阅读 **[DEVELOPERS.md](DEVELOPERS.md)**

---

## 🏗️ 架构设计

### 系统架构图

```
┌──────────────────────────────────────────────────────┐
│                  用户交互层                           │
│   ┌─────────────────┐      ┌─────────────────┐      │
│   │   Web前端       │      │   HTTP客户端    │      │
│   │  (Dioxus WASM)  │      │  (Python/cURL)  │      │
│   └────────┬────────┘      └────────┬────────┘      │
└────────────┼─────────────────────────┼───────────────┘
             │ HTTP                    │ HTTP
             ▼                         ▼
┌──────────────────────────────────────────────────────┐
│                  infer-server                         │
│  • OpenAI兼容API（/v1/chat/completions）             │
│  • 聊天模板处理（Llama3格式）                         │
│  • ZMQ客户端（DEALER socket）                        │
│  • 健康检查 & 性能指标                                │
└──────────────────┬───────────────────────────────────┘
                   │ ZeroMQ IPC (MessagePack)
                   ▼
┌──────────────────────────────────────────────────────┐
│                  infer-protocol                       │
│  • 请求/响应消息定义                                  │
│  • MessagePack序列化                                  │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│                  infer-engine                         │
│  • 推理引擎进程（独立GPU进程）                        │
│  • 请求队列 & 调度                                    │
│  • ZMQ服务器（ROUTER socket）                        │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│                  infer-core                           │
│  • 模型加载（safetensors）                            │
│  • Transformer实现（Llama3）                          │
│  • 算子库（CPU/CUDA）                                 │
│  • 张量系统（零拷贝）                                 │
│  • 内存管理（池化分配器）                             │
└──────────────────────────────────────────────────────┘
```

### 为什么采用进程分离架构？

1. **隔离性**: GPU推理进程与HTTP服务器独立，崩溃不影响对方
2. **可靠性**: 重启服务器无需重新加载模型（节省30秒+启动时间）
3. **可扩展性**: 单个引擎可服务多个服务器实例
4. **性能**: ZeroMQ IPC延迟仅10-50μs，MessagePack比JSON小5-10x

### 项目结构

```
RustInfer/
├── crates/
│   ├── infer-protocol/    # 通信协议定义（MessagePack）
│   ├── infer-engine/      # 独立推理引擎进程
│   ├── infer-core/        # 核心推理库
│   │   ├── base/          # 内存管理、分配器
│   │   ├── tensor/        # 张量系统（零拷贝）
│   │   ├── op/            # 算子库（CPU/CUDA）
│   │   ├── model/         # 模型实现（Llama3）
│   │   └── cuda/          # CUDA集成
│   ├── infer-server/      # HTTP API服务器（Axum）
│   │   ├── api/           # OpenAI兼容端点
│   │   ├── chat/          # 聊天模板
│   │   └── zmq_client.rs  # ZMQ客户端
│   └── infer-frontend/    # Web UI（Dioxus WASM）
├── DEVELOPERS.md          # 开发者文档（架构深度解析）
├── README.md              # 本文件
└── Cargo.toml             # 工作区配置
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install clang libclang-dev pkg-config libssl-dev

# OpenBLAS（CPU后端）
sudo apt-get install libopenblas-dev

# 或使用Conda
conda install conda-forge::libclang anaconda::openssl
```

### 2. 克隆仓库

```bash
git clone https://github.com/Vinci-hit/RustInfer.git
cd RustInfer
```

### 3. 构建项目

```bash
# CPU版本
cargo build --release

# CUDA版本（需要CUDA toolkit）
cargo build --release --features cuda
```

**注意**: 修改 `crates/infer-core/build.rs` 中的计算能力flag以适配你的GPU（默认sm_89用于4070Ti super）

### 4. 运行测试

```bash
# 基础测试
cargo test

# CUDA性能测试
cargo test test_llama3_cuda_performance --release -- --nocapture --ignored

# CPU推理测试
cargo test test_llama3_cpu_loading_and_generation --release -- --nocapture --ignored
```

### 5. 启动服务

#### 方式1: 完整服务（推荐）

```bash
# 终端1: 启动engine 和 server
sh scripts/start_distributed.sh

# 终端2: 启动Web前端（可选）
cd crates/infer-frontend
dx serve --port 3000
```

#### 方式2: 使用API

```bash
# 测试API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "解释什么是Rust语言"}],
    "stream": false
  }'

# Python OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

#### 方式3: 作为Rust库

```toml
[dependencies]
infer-core = { path = "path/to/RustInfer/crates/infer-core", features = ["cuda"] }
```

```rust
use infer_core::model::llama3::Llama3;
use infer_core::base::DeviceType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Llama3::new(
        "/path/to/model",
        DeviceType::Cuda(0),
        false  // 非量化模型
    )?;

    let (text, tokens, prefill_ms, decode_ms, _) =
        model.generate("你好", 100, false)?;

    println!("生成: {}", text);
    println!("性能: prefill {}ms, decode {}ms, {} tokens",
             prefill_ms, decode_ms, tokens);
    Ok(())
}
```

---

## 📦 支持的模型

### Llama 3 系列

| 模型 | 参数量 | 推荐设备 | 测试状态 |
|------|--------|----------|----------|
| Llama-3.2-1B-Instruct | 1B | CPU / GPU | ✅ 完全支持 |
| Llama-3.2-3B | 3B | GPU | 🔄 理论支持 （其实不支持，算子需要调）|
| Llama-3.1-8B | 8B | GPU (8GB+) | 🔄 理论支持（其实不支持，算子需要调） |

**下载地址**:
- [Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
- [Meta Llama系列](https://huggingface.co/meta-llama)

**支持格式**:
- ✅ SafeTensors (.safetensors)
- ✅ 分片模型 (model.safetensors.index.json)
- ✅ HuggingFace Tokenizer (tokenizer.json)

---

## 📊 性能基准与版本演进

### 性能提升（vs v0.1.0）

> **测试环境**: H200, Llama-3.2-1B-Instruct, Batch Size=1

| 指标 | v0.1.0 | v0.2.0 | 提升 |
|------|--------|--------|------|
| **Prefill 吞吐量** | ~355 tok/s | ~1052 tok/s | **3x ⬆️** |
| **Decode 吞吐量** | ~220 tok/s | ~436 tok/s | **2x ⬆️** |
| **模型加载时间** | ~15 秒 | ~5 秒 | **3x ⬆️** |
| **显存占用** | ~12GB (FP32) | ~6GB (BF16) | **50% ⬇️** |

### 版本历史

#### v0.4.0 (当前) - 架构升级
**发布日期**: 2026-01

**核心改进**:
- ✨ **进程分离架构**: 将推理引擎拆分为独立进程
  - 新增 `infer-engine` crate（独立GPU进程）
  - 新增 `infer-protocol` crate（通信协议模板）
  - 新增 `infer-server` ZMQ客户端
  - 实现 ROUTER-DEALER socket通信
- 🚀 **ZeroMQ IPC通信**: MessagePack序列化（5-10x小于JSON）
- 🔧 **可靠性提升**: 服务器重启无需重载模型
- 📈 **可扩展性**: 单引擎支持多服务器实例

**架构优势**:
- 隔离性：GPU进程与HTTP服务器独立
- 可维护性：组件职责清晰分离
- 灵活性：可独立扩展和部署

#### v0.3.0 - CUDA Graph优化
**发布日期**: 2026-01

**核心改进**:
- ⚡ **CUDA Graph捕获与回放**
  - Decode阶段自动捕获计算图
  - Kernel启动开销降低
  - 降低GPU空闲时间
- 🔧 实现 `CudaConfig` 管理图生命周期
- 📊 Decode延迟进一步降低

**技术细节**:
- 首次迭代捕获完整计算图
- 后续迭代直接回放（单次GPU调用）
- 消除重复的kernel启动开销

#### v0.2.0 - 性能突破
**发布日期**: 2026-01

**核心改进**:
- 🚀 **BF16混合精度推理**
  - Prefill吞吐量: 355 → 1052 tok/s (**3x**)
  - Decode吞吐量: 220 → 436 tok/s (**2x**)
  - 显存占用: 4GB → 2GB (**50%**)

**优化技术**:
- Flash Attention GQA实现
- 算子融合（SwiGLU等）
- cuBLASLt自动调优

#### v0.1.0 - 初始版本
**发布日期**: 2025-10

**基础功能**:
- ✅ Llama3模型完整推理
- ✅ CPU和CUDA双后端
- ✅ KV缓存管理
- ✅ OpenAI兼容API
- ✅ 基础Transformer算子
- ✅ F32推理支持

**性能基线**:
- Prefill: ~355 tok/s
- Decode: ~220 tok/s
- 加载时间: ~15s
- 显存占用: ~12GB (FP32)

---

## ⚡ 性能优化技术

### 已实现的优化

1. **CUDA内存池化** (`/crates/infer-core/src/base/allocator.rs`)
   - 分配延迟: 800µs → 1µs
   - 线程安全并发访问（DashMap）
   - 双层池策略（小块first-fit，大块best-fit）

2. **零拷贝模型加载** (`/crates/infer-core/src/model/loader.rs`)
   - mmap直接映射文件
   - 无需反序列化（100x加速）
   - 安全的生命周期扩展

3. **Workspace预分配** (`/crates/infer-core/src/model/llama3.rs`)
   - 预分配最大尺寸缓冲区
   - 推理循环零内存分配
   - HashMap管理命名缓冲区

4. **CUDA Graph捕获** (`/crates/infer-core/src/cuda/config.rs`)
   - 首次迭代捕获计算图
   - 后续迭代回放图（10-100x加速）
   - 消除kernel启动开销

5. **Flash Attention** (`/crates/infer-core/src/op/kernels/cuda/flash_attn_gqa/`)
   - 分块注意力计算
   - 在线softmax
   - 减少3x内存访问

6. **算子融合**
   - SwiGLU: gate + silu + multiply单kernel
   - 减少kernel启动和内存往返

7. **BF16混合精度**
   - GPU使用BFloat16
   - 2x内存带宽
   - FP32累加器保证精度

---

## ⚠️ 当前限制

### 已实现 ✅
- [x] Llama3模型完整推理
- [x] 进程分离架构（ZeroMQ IPC）
- [x] KV缓存管理
- [x] CPU和CUDA双后端
- [x] F32和BF16数据类型
- [x] OpenAI兼容API
- [x] 流式响应（SSE）
- [x] CUDA Graph优化
- [x] Flash Attention GQA

### 待实现 🔄

**高优先级**:
- 采样器：仅argmax，缺少temperature/top-p/top-k
- 连续批处理：目前串行处理请求
- PagedAttention：固定KV缓存大小
- 量化支持：INT8/INT4

**中优先级**:
- 多模型架构支持
- 部分算子CUDA优化不足
- 错误处理改进（减少unwrap）

**低优先级**:
- 自定义停止序列
- Token概率输出
- Function calling
- API认证机制

详细技术实现指南请参阅 **[DEVELOPERS.md](DEVELOPERS.md)**

---

## 🛠️ 开发指南

### 代码规范

```bash
# 格式化
cargo fmt

# Lint检查
cargo clippy -- -D warnings

# 文档生成
cargo doc --no-deps --open
```

### 添加新算子

请参阅 [DEVELOPERS.md](DEVELOPERS.md) 中的详细模板和示例。

关键步骤:
1. 实现 `Op` trait
2. CPU和CUDA双后端
3. 编写CUDA kernel（可选）
4. 添加单元测试

### 添加新模型

参考 `/crates/infer-core/src/model/llama3.rs` 实现:
1. 定义配置结构
2. 实现层组合
3. Workspace管理
4. 两阶段推理（prefill/decode）

完整指南: [DEVELOPERS.md](DEVELOPERS.md)

---

## 🧪 测试

```bash
# 所有测试
cargo test

# 性能基准
cargo test --release -- --nocapture --ignored

# 特定crate
cd crates/infer-core && cargo test
```

### 测试覆盖

| 模块 | 单元测试 | 集成测试 | 性能测试 |
|------|----------|----------|----------|
| infer-core | ✅ | ✅ | ✅ |
| infer-engine | ✅ | ⚠️ | ⚠️ |
| infer-server | ⚠️ | ⚠️ | ❌ |
| infer-frontend | ❌ | ❌ | ❌ |

---

## 🤝 贡献指南

欢迎贡献！在提交PR前请确保:

- [ ] 通过 `cargo fmt` 格式化
- [ ] 通过 `cargo clippy` 检查
- [ ] 所有测试通过
- [ ] 添加必要的文档
- [ ] 更新相关文档

**优先贡献方向**:
- 🐛 Bug修复
- ⚡ 性能优化
- 📝 文档改进
- ✨ 新算子/模型
- 🧪 测试覆盖

详细贡献指南: [DEVELOPERS.md](DEVELOPERS.md)

---

## 📚 文档导航

- **[DEVELOPERS.md](DEVELOPERS.md)** - 完整开发者文档
  - 设计哲学详解
  - 架构深度剖析
  - 内存管理系统
  - 性能优化技术
  - 贡献者指南（含完整代码模板）

- **学习资源**
  - [Rust官方文档](https://doc.rust-lang.org/)
  - [CUDA编程指南](https://docs.nvidia.com/cuda/)
  - [Axum Web框架](https://docs.rs/axum/)

---

## 📄 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

**灵感来源**:
- [KuiperLLama](https://github.com/zjhellofss/KuiperLLama) - 课程项目
- [vLLM](https://github.com/vllm-project/vllm) - 推理引擎设计

**技术栈**:
- 🦀 Rust - 内存安全与零成本抽象
- ⚡ CUDA - GPU加速
- 🌐 Axum + Dioxus - 现代Web
- 📦 ZeroMQ - 高性能IPC
- 🎯 HuggingFace - 模型生态

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐ Star！**

Made with ❤️ and 🦀 Rust

[GitHub](https://github.com/Vinci-hit/RustInfer) • [Issues](https://github.com/Vinci-hit/RustInfer/issues)

</div>
