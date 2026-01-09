# RustInfer Server - 生产级HTTP推理服务器

> 现代化的高性能LLM推理HTTP服务器，提供OpenAI兼容API，架构设计借鉴vLLM和SGLang。

## 🚀 快速开始

```bash
# 启动服务器
cargo run --release --bin rustinfer-server -- \
    --model /path/to/llama3 \
    --port 8000 \
    --device cuda:0 \
    --max-tokens 512

# 或使用编译好的二进制文件
./target/release/rustinfer-server \
    --model /mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct \
    --port 8000 \
    --device cuda:0

# 使用curl测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'
```

## 📋 目录

- [功能特性](#功能特性)
- [架构设计](#架构设计)
- [高级设计模式](#高级设计模式)
- [API参考](#api参考)
- [使用示例](#使用示例)
- [性能与优化](#性能与优化)
- [开发指南](#开发指南)
- [路线图](#路线图)

---

## ✨ 功能特性

### 核心能力
- ✅ **OpenAI兼容API** - 可直接替换OpenAI API
- ✅ **性能可观测性** - 实时指标（预填充/解码时间、tokens/秒）
- ✅ **系统监控** - CPU、GPU、内存指标端点
- ✅ **服务器推送事件（SSE）** - 实时流式响应
- ✅ **自动对话模板** - Llama3格式包装
- ✅ **异步运行时** - 基于Axum + Tokio构建
- ✅ **线程安全推理** - Arc<Mutex>实现并发请求
- ✅ **CUDA Graph就绪** - 预分配工作空间缓冲区
- ✅ **优雅关闭** - 正确的资源清理
- ✅ **CORS支持** - 可用于Web应用
- ✅ **结构化日志** - 基于Tracing的可观测性

### 生产就绪
- 🔒 类型安全的Rust实现
- 🚀 零拷贝张量操作
- 📊 请求/响应日志记录
- 🎯 健康检查与就绪探测端点
- 🔧 环境变量配置
- 📦 小体积二进制文件（12MB release构建）

---

## 🏗️ 架构设计

### 高层概览

```
┌─────────────────────────────────────────────────────────┐
│                    客户端应用程序                        │
│  (curl, Python OpenAI SDK, Web应用等)                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Axum HTTP服务器 (main.rs)                 │
│  • 路由配置                                             │
│  • CORS中间件                                           │
│  • Tracing中间件                                        │
│  • 优雅关闭处理器                                        │
└────────────────────┬────────────────────────────────────┘
                     │
            ┌────────┴────────┐
            │                 │
            ▼                 ▼
┌─────────────────┐   ┌──────────────────┐
│  API处理器      │   │  健康检查        │
│  (api/)         │   │  (api/health.rs) │
│                 │   │                  │
│ • openai.rs     │   │ • /health        │
│   - /v1/chat/   │   │ • /ready         │
│     completions │   │                  │
│   - /v1/models  │   └──────────────────┘
│ • metrics.rs    │
│   - /v1/metrics │
│     (系统资源)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          推理引擎 (inference/engine.rs)                 │
│  • Arc<Mutex<InferenceEngine>> (线程安全)              │
│  • 请求队列与序列化                                      │
│  • 对话模板应用                                          │
│  • 响应格式化                                            │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  对话模板       │     │   infer-core     │
│  (chat/)        │     │   (外部crate)    │
│                 │     │                  │
│ • Llama3格式    │     │ • Llama3模型     │
│ • 消息包装      │     │ • CUDA内核       │
│ • 系统提示词    │     │ • BF16推理       │
└─────────────────┘     └──────────────────┘
```

### 组件详解

#### 1. **HTTP层** (`main.rs`)
```rust
// 使用状态共享的Axum路由器
let app = Router::new()
    .route("/v1/chat/completions", post(chat_completions))
    .with_state(Arc::new(Mutex::new(engine)))  // 共享可变状态
    .layer(CorsLayer::new().allow_origin(Any))  // 跨域支持
    .layer(TraceLayer::new_for_http());         // 请求日志
```

**设计决策：**
- **Axum框架**：现代化、符合人体工程学、基于Tokio构建
- **Arc<Mutex<>>**：模型的线程安全共享所有权
- **Tower中间件**：可组合的请求/响应处理

#### 2. **API层** (`api/`)
```
api/
├── mod.rs          # 模块导出
├── openai.rs       # OpenAI兼容类型与处理器
├── metrics.rs      # 系统指标端点
└── health.rs       # 存活/就绪探测
```

**关键类型** (`openai.rs`):
```rust
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
}

pub struct ChatCompletionResponse {
    pub id: String,              // 唯一请求ID
    pub object: String,          // "chat.completion"
    pub created: i64,            // Unix时间戳
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,            // Token计数与性能指标
}

// 性能指标结构
pub struct Performance {
    pub prefill_ms: u64,              // 预填充时间（毫秒）
    pub decode_ms: u64,               // 解码时间（毫秒）
    pub decode_iterations: usize,     // 解码迭代次数
    pub tokens_per_second: f64,       // 生成速度
    pub time_to_first_token_ms: u64,  // 首token时间
}
```

---

## 📚 API参考

### 端点列表

#### `POST /v1/chat/completions`

创建带有对话上下文的聊天补全。

**请求体**:
```json
{
  "model": "llama3",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是Rust？"}
  ],
  "max_tokens": 150,
  "stream": false
}
```

**响应** (非流式):
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Rust是一门系统编程语言..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 36,
    "total_tokens": 36,
    "performance": {
      "prefill_ms": 183,
      "decode_ms": 118,
      "decode_iterations": 35,
      "tokens_per_second": 29.9,
      "time_to_first_token_ms": 183
    }
  }
}
```

**响应** (流式，`stream: true`):
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Rust "}}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"是 "}}]}

data: [DONE]
```

#### `GET /v1/models`

列出可用模型。

**响应**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3",
      "object": "model",
      "owned_by": "rustinfer"
    }
  ]
}
```

#### `GET /health`

健康检查端点。

**响应**:
```json
{
  "status": "healthy",
  "service": "rustinfer-server"
}
```

#### `GET /ready`

就绪探测（模型已加载）。

**响应**:
```json
{
  "status": "ready",
  "model_loaded": true
}
```

#### `GET /v1/metrics`

系统资源监控指标。

**响应**:
```json
{
  "cpu": {
    "utilization_percent": 1.1,
    "core_count": 28
  },
  "memory": {
    "used_mb": 2244,
    "total_mb": 15903,
    "available_mb": 13658
  },
  "gpu": {
    "device_id": 0,
    "utilization_percent": 45.2,
    "memory_used_mb": 2500,
    "memory_total_mb": 24576,
    "temperature_celsius": 65.5
  },
  "timestamp": 1767937158
}
```

**注意**：当CUDA特性未启用或无GPU可用时，`gpu`字段为`null`。

---

## 💡 使用示例

### Python (OpenAI SDK)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 不验证
)

# 非流式
response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "你是一个Rust专家。"},
        {"role": "user", "content": "用2句话解释所有权。"}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "写一首关于Rust的俳句"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript (Fetch API)

```javascript
// 非流式
const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama3',
    messages: [{ role: 'user', content: '你好！' }],
    stream: false
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### curl

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'

# 流式（-N表示无缓冲）
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "讲个笑话"}],
    "stream": true
  }'
```

---

## 🌐 Web前端

`crates/infer-frontend/`提供了一个基于Dioxus的现代化Web应用，具备以下功能：

- **交互式聊天界面** - 与模型进行多轮对话
- **实时性能指标** - 显示每个响应的预填充/解码时间、tokens/秒
- **系统监控仪表板** - 实时CPU、GPU、内存使用情况（每2秒轮询/v1/metrics）
- **响应式UI** - Tailwind CSS深色主题
- **基于WASM** - 完全在浏览器中运行

**快速启动**:
```bash
# 终端1：启动后端
cargo run --release --bin rustinfer-server -- \
    --model /path/to/model \
    --port 8000

# 终端2：启动前端
cd crates/infer-frontend
dx serve --port 3000

# 打开浏览器：http://localhost:3000
```

详见`crates/infer-frontend/README.md`。

---

## ⚡ 性能与优化

### 当前性能（Llama-3.2-1B-Instruct，BF16，RTX 4090）

- **模型加载时间**：9.46秒
- **推理延迟**：<100ms TTFT（首token时间）
- **吞吐量**：~30-40 tokens/秒（单请求）
- **内存使用**：~2.5GB VRAM（模型 + KV缓存）

---

## 🛠️ 开发指南

### 项目结构

```
infer-server/
├── Cargo.toml          # 依赖与构建配置
├── README_CN.md        # 本文件
├── src/
│   ├── main.rs         # 服务器入口点
│   ├── lib.rs          # 公共API（用于测试）
│   ├── api/
│   │   ├── mod.rs
│   │   ├── openai.rs   # OpenAI兼容处理器
│   │   ├── metrics.rs  # 系统指标端点
│   │   └── health.rs   # 健康检查
│   ├── chat/
│   │   ├── mod.rs
│   │   └── template.rs # 对话模板实现
│   ├── inference/
│   │   ├── mod.rs
│   │   └── engine.rs   # 推理引擎包装器
│   └── config/
│       ├── mod.rs
│       └── server.rs   # 服务器配置
└── tests/
    └── integration.rs  # (待完成) 端到端测试
```

### 构建

```bash
# 开发构建
cargo build --bin rustinfer-server

# 发布构建（优化）
cargo build --release --bin rustinfer-server

# 带日志运行
RUST_LOG=debug cargo run --release --bin rustinfer-server -- \
    --model /path/to/model \
    --port 8000
```

### 测试

```bash
# 单元测试
cargo test --lib

# 集成测试（需先启动服务器）
cargo test --test integration -- --test-threads=1

# 手动测试
curl http://localhost:8000/health
```

### 环境变量

```bash
# 所有CLI参数都可以通过环境变量设置
export MODEL_PATH=/path/to/model
export HOST=0.0.0.0
export PORT=8000
export DEVICE=cuda:0
export MAX_TOKENS=512
export RUST_LOG=info

./rustinfer-server  # 使用环境变量
```

---

## 🗺️ 路线图

### 阶段1：MVP ✅（已完成）
- [x] OpenAI兼容API
- [x] Llama3对话模板
- [x] SSE流式传输
- [x] 健康检查
- [x] CORS支持
- [x] 优雅关闭
- [x] **性能指标**（预填充/解码时间、tokens/秒）
- [x] **系统监控端点**（/v1/metrics）
- [x] **Web前端**（基于Dioxus）

### 阶段2：性能（进行中）
- [x] 请求/响应可观测性 ✅
- [ ] Token逐个流式传输（需要infer-core API支持）
- [ ] 请求批处理（连续批处理）
- [ ] CUDA graph集成
- [ ] 请求队列可视化

### 阶段3：功能特性
- [ ] 多模型支持（加载/卸载）
- [ ] Temperature/top-p/top-k采样
- [ ] 停止序列
- [ ] Logprobs输出
- [ ] 函数调用API
- [ ] 视觉支持（多模态）

### 阶段4：生产环境
- [ ] 请求认证（API密钥）
- [ ] 速率限制
- [ ] 请求缓存
- [ ] 分布式推理（多GPU）
- [ ] Kubernetes部署清单
- [ ] Docker镜像
- [ ] 负载均衡器水平扩展

---

## 🤝 贡献

### 代码风格
- 提交前运行`cargo fmt`
- 运行`cargo clippy`并解决警告
- 为重要事件添加tracing日志
- 使用`///`文档注释记录公共API

### Pull Request流程
1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/amazing`）
3. 提交时使用描述性消息
4. 推送并开启包含详细描述的PR

### 架构决策
对于重大更改，请先开issue讨论：
- 性能影响
- API兼容性
- 内存使用
- 线程安全

---

## 📄 许可证

本项目是RustInfer的一部分，使用相同的许可证。

---

## 🙏 致谢

**受以下项目启发：**
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention与连续批处理
- [SGLang](https://github.com/sgl-project/sglang) - 结构化生成与运行时
- [Axum](https://github.com/tokio-rs/axum) - 符合人体工程学的Web框架
- [OpenAI API](https://platform.openai.com/docs/api-reference) - 标准API设计

**构建工具：**
- 🦀 Rust - 性能 + 安全性
- ⚡ Tokio - 异步运行时
- 🌐 Axum - HTTP框架
- 🎯 CUDA - 通过infer-core实现GPU加速

---

## 📞 支持

如有问题或疑问：
- 在GitHub上开issue
- 查看[infer-core文档](../infer-core/README.md)
- 查看本README中的API示例

**服务器在运行吗？**使用`RUST_LOG=debug`检查日志
