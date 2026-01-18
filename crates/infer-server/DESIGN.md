# Infer Server 设计文档

> RustInfer 分布式推理系统的 HTTP 网关层

## 目录

- [设计哲学](#设计哲学)
- [服务边界](#服务边界)
- [架构概览](#架构概览)
- [核心模块](#核心模块)
- [API 规范](#api-规范)
- [通信协议](#通信协议)
- [状态管理](#状态管理)
- [扩展指南](#扩展指南)

---

## 设计哲学

### 核心原则

1. **职责单一**：Server 层专注于 HTTP 协议处理和请求路由，不涉及推理计算
2. **轻量转发**：Server 与 Scheduler 之间仅传输 Token ID，避免传输大文本数据
3. **流式优先**：所有推理响应采用流式输出，最小化首字延迟 (TTFT)
4. **协议兼容**：提供 OpenAI 兼容的 API，降低迁移成本
5. **状态无感知**：HTTP 层保持无状态，所有状态由 Scheduler 和 Worker 管理
6. **可观测性**：内置完整的监控指标和健康检查接口

### 设计权衡

| 决策 | 优势 | 代价 |
|------|------|------|
| Token ID 传输 | 减少网络带宽，提高并发 | Server 需要维护 Tokenizer |
| 流式 SSE | 低延迟，实时反馈 | 连接管理复杂 |
| ZeroMQ | 高性能，异步 I/O | 学习曲线，调试困难 |
| MessagePack | 紧凑序列化，跨语言 | 调试可读性差 |

---

## 服务边界

### 职责范围

#### ✅ Server 负责

- HTTP/HTTPS 协议处理
- 请求参数验证和转换
- 文本 Token 化 (Tokenize / Detokenize)
- 对话模板应用 (Chat Templates)
- 流式响应生成 (Server-Sent Events)
- 请求路由和负载均衡
- 监控指标收集
- 健康检查接口

#### ❌ Server 不负责

- 推理计算 (Inference)
- 请求调度 (Scheduling)
- KV Cache 管理
- 模型权重加载
- GPU 资源分配
- 请求优先级管理
- Batch 合并优化

### 与其他组件的交互

```
┌─────────────────┐     HTTP/SSE     ┌─────────────────┐
│   HTTP Clients  │ <──────────────> │  Infer Server   │
└─────────────────┘                  └────────┬────────┘
                                              │
                                         ZeroMQ
                                              │
                                     (Token ID only)
                                              │
                                     ┌────────▼────────┐
                                     │  Infer Scheduler│
                                     └────────┬────────┘
                                              │
                                         Actor IPC
                                              │
                                     ┌────────▼────────┐
                                     │  Infer Workers │
                                     │   (CUDA)        │
                                     └─────────────────┘
```

### 数据流边界

```
Client                    Server                  Scheduler                   Worker
────                      ────                     ───────                     ─────
[Text]                    [Text]
                          │
                      Tokenize
                          │
                      [Token IDs] ──────────────▶ [Token IDs]
                                                   │
                                                Schedule
                                                   │
                                                   ├──────▶ [Token IDs] (Batch)
                                                   │
                                                   ├──────▶ [Token IDs] (Batch)
                                                   │
                                                   └──────▶ [Token IDs] (Batch)
                                                   │
                                               Inference
                                                   │
                                            [Token ID] ───────────────▶ [Token ID]
                                                                           │
                                                                      Generate
                                                                           │
                                                                       [Token ID] ──┐
                                                                                   │
[Token] ◀─── Detokenize ◀─── [Token ID] ◀─────────────────────────────────────┘
```

---

## 架构概览

### 目录结构

```
crates/infer-server/
├── src/
│   ├── main.rs              # 服务入口和启动逻辑
│   ├── lib.rs               # 库接口导出
│   ├── config.rs            # 配置管理
│   ├── state.rs             # 全局状态管理
│   │
│   ├── backend/             # 后端通信层
│   │   └── zmq_client.rs    # ZeroMQ 通信循环
│   │
│   ├── processor/           # 文本处理层
│   │   └── tokenizer.rs     # Tokenizer 和流式解码
│   │
│   ├── http/                # HTTP 层
│   │   ├── handler.rs       # HTTP 请求处理器
│   │   ├── request.rs       # 请求结构定义
│   │   └── response.rs      # 响应结构定义
│   │
│   ├── api/                 # API 实现
│   │   ├── openai.rs        # OpenAI 兼容 API
│   │   ├── health.rs        # 健康检查
│   │   └── metrics.rs       # 监控指标
│   │
│   └── chat/                # 对话模板
│       └── template.rs      # Chat Template 实现
│
└── Cargo.toml               # 依赖定义
```

### 启动流程

```
main()
  │
  ├─▶ 加载配置 (ServerConfig::from_env)
  │
  ├─▶ 初始化日志 (tracing)
  │
  ├─▶ 加载 Tokenizer
  │
  ├─▶ 创建 AppState
  │
  ├─▶ 启动 ZeroMQ 后台线程
  │
  ├─▶ 构建 HTTP 路由
  │
  └─▶ 启动 HTTP 服务器
```

### 运行时架构

```
HTTP Server Thread          ZMQ Background Thread
      │                             │
      │    mpsc::unbounded         │
      ├────────────────────────▶   │
      │  SchedulerCommand          │
      │                             │
      │                             │  ZMQ Socket
      │                             ├─────────────▶ Scheduler
      │                             │
      │                             │  SchedulerOutput
      │                             ◀─────────────┤
      │                             │
      │    mpsc::unbounded         │
      │  SchedulerOutput           │
      ◀────────────────────────────┘
      │
      │  SSE Stream
      ◀───────────────────▶ Client
```

---

## 核心模块

### 1. Config 模块 (`config.rs`)

**职责**：管理服务器配置，支持环境变量加载

**核心结构**：

```rust
pub struct ServerConfig {
    pub http: HttpConfig,           // HTTP 服务器配置
    pub scheduler: SchedulerConfig, // Scheduler 连接配置
    pub model: ModelConfig,          // 模型配置
    pub log: LogConfig,              // 日志配置
}

pub struct HttpConfig {
    pub port: u16,                   // HTTP 端口
    pub cors_enabled: bool,          // 是否启用 CORS
}

pub struct SchedulerConfig {
    pub endpoint: String,            // ZeroMQ 端点地址
    pub timeout_ms: u64,             // 超时时间
}

pub struct ModelConfig {
    pub model_name: String,          // 模型名称
    pub tokenizer_path: String,      // Tokenizer 路径
}
```

**关键方法**：
- `ServerConfig::from_env()` - 从环境变量加载配置
- `ServerConfig::validate()` - 配置验证

### 2. State 模块 (`state.rs`)

**职责**：维护全局共享状态，管理请求生命周期

**核心结构**：

```rust
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub scheduler_tx: mpsc::UnboundedSender<SchedulerCommand>,
    pub request_map: Arc<RwLock<RequestMap>>,
}

pub struct RequestMap {
    map: HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>,
}
```

**关键方法**：
- `AppState::register_request()` - 注册新请求
- `AppState::unregister_request()` - 清理请求
- `AppState::get_request()` - 获取请求通道
- `AppState::active_count()` - 获取活跃请求数

**线程安全保证**：
- `config` 和 `tokenizer` 使用 `Arc` 共享不可变数据
- `request_map` 使用 `Arc<RwLock>` 保护并发访问

### 3. Backend 模块 (`backend/zmq_client.rs`)

**职责**：处理与 Scheduler 的 ZeroMQ 双向通信

**核心逻辑**：

```rust
pub async fn run_zmq_loop(
    rx: mpsc::UnboundedReceiver<SchedulerCommand>,
    state: Arc<AppState>,
) -> Result<()> {
    // 创建 ZeroMQ DEALER socket
    let context = zmq::Context::new();
    let socket = context.socket(zmq::DEALER)?;

    // 连接到 Scheduler
    socket.connect(&config.scheduler.endpoint)?;

    loop {
        select! {
            // 接收来自 HTTP Handler 的命令
            cmd = rx.recv() => {
                send_to_scheduler(&socket, cmd?)?;
            }

            // 接收来自 Scheduler 的输出
            output = recv_from_scheduler(&socket) => {
                route_to_http_handler(output, &state).await?;
            }
        }
    }
}
```

**通信特点**：
- 在独立线程中运行，避免阻塞 HTTP 处理
- 使用 `select!` 宏实现双向通信
- MessagePack 序列化优化传输效率

### 4. Processor 模块 (`processor/tokenizer.rs`)

**职责**：文本编码和解码，处理流式输出

**核心结构**：

```rust
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

pub struct StreamDecoder {
    decoder: TokenStreamDecoder,
    eos_token_id: Option<u32>,
}
```

**关键方法**：

```rust
impl TokenizerWrapper {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // 文本 → Token IDs
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Token IDs → 文本
    }

    pub fn stream_decoder(&self) -> StreamDecoder {
        // 创建流式解码器
    }
}

impl StreamDecoder {
    pub fn decode(&mut self, token: u32) -> Option<String> {
        // 增量解码单个 Token，处理 UTF-8 边界
    }
}
```

**UTF-8 边界处理**：
- 使用 `TokenStreamDecoder` 正确处理多字节字符
- 确保流式输出不会出现乱码

### 5. HTTP Handler 模块 (`http/handler.rs`)

**职责**：处理 HTTP 请求，生成 SSE 流式响应

**核心逻辑**：

```rust
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    req: Json<ChatCompletionRequest>,
) -> Result<Response<Body>> {
    // 1. 生成 request_id
    let request_id = generate_request_id();

    // 2. 应用对话模板
    let prompt = apply_chat_template(&req.messages)?;

    // 3. Tokenize
    let tokens = state.tokenizer.encode(&prompt)?;

    // 4. 创建响应通道
    let (tx, mut rx) = mpsc::unbounded_channel();
    state.register_request(request_id.clone(), tx)?;

    // 5. 构造 SchedulerCommand
    let command = SchedulerCommand::AddRequest(InferRequest {
        request_id: request_id.clone(),
        tokens,
        sampling_params: req.to_sampling_params(),
    });

    // 6. 发送到 Scheduler
    state.scheduler_tx.send(command)?;

    // 7. 创建 SSE 流
    let stream = async_stream::stream! {
        let mut decoder = state.tokenizer.stream_decoder();

        while let Some(output) = rx.recv().await {
            match output {
                SchedulerOutput::Step(step) => {
                    // 增量解码
                    if let Some(text) = decoder.decode(step.token) {
                        yield Event::default()
                            .json_data(Chunk::new(text))
                            .unwrap();
                    }
                }
                SchedulerOutput::Finish(finish) => {
                    // 发送完成消息
                    yield Event::default()
                        .json_data(Chunk::finish(finish))
                        .unwrap();
                    break;
                }
                SchedulerOutput::Error(err) => {
                    // 发送错误消息
                    yield Event::default()
                        .json_data(Chunk::error(err))
                        .unwrap();
                    break;
                }
            }
        }

        // 清理
        state.unregister_request(&request_id);
    };

    Ok(Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::from_stream(stream))?)
}
```

**请求生命周期**：

```
HTTP Request
    │
    ├─▶ 验证参数
    │
    ├─▶ 应用对话模板
    │
    ├─▶ Tokenize
    │
    ├─▶ 注册请求 (create channel)
    │
    ├─▶ 发送到 Scheduler
    │
    ├─▶ SSE Stream Loop
    │   │
    │   ├─▶ 接收 SchedulerOutput
    │   │
    │   ├─▶ Detokenize
    │   │
    │   └─▶ 发送 SSE Event
    │
    └─▶ 清理请求
```

### 6. API 模块

#### OpenAI API (`api/openai.rs`)

**职责**：提供 OpenAI 兼容的 API 接口

**端点**：
- `POST /v1/chat/completions` - 聊天完成
- `GET /models` - 模型列表

**兼容性**：
- 完全兼容 OpenAI API 规范
- 支持所有采样参数 (temperature, top_p, max_tokens, etc.)

#### Health API (`api/health.rs`)

**职责**：服务健康检查

**端点**：
- `GET /health` - 基础健康检查

**检查项**：
- HTTP 服务状态
- Scheduler 连接状态
- Tokenizer 加载状态

#### Metrics API (`api/metrics.rs`)

**职责**：系统监控指标

**端点**：
- `GET /metrics` - Prometheus 格式指标

**指标类别**：
- HTTP 请求指标 (请求数, 延迟, 错误率)
- 活跃请求数
- CPU / 内存 / GPU 使用率
- 缓存命中率

---

## API 规范

### OpenAI 兼容 API

#### POST /v1/chat/completions

**请求格式**：

```json
{
  "model": "llama3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1024,
  "stream": true
}
```

**参数说明**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型名称 |
| messages | array | 是 | 对话消息列表 |
| temperature | number | 否 | 采样温度 (0-2)，默认 0.7 |
| top_p | number | 否 | 核采样阈值 (0-1)，默认 0.9 |
| max_tokens | integer | 否 | 最大生成 Token 数，默认 1024 |
| stream | boolean | 否 | 是否流式输出，默认 true |
| stop | array | 否 | 停止字符串列表 |
| presence_penalty | number | 否 | 存在惩罚系数，默认 0 |
| frequency_penalty | number | 否 | 频率惩罚系数，默认 0 |

**响应格式 (流式)**：

```
data: {"id":"123","object":"chat.completion.chunk","created":1700000000,"model":"llama3-8b","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}]}

data: {"id":"123","object":"chat.completion.chunk","created":1700000000,"model":"llama3-8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**完成原因 (finish_reason)**：

| 值 | 说明 |
|----|------|
| stop | 正常完成 (达到停止条件) |
| length | 达到 max_tokens |
| content_filter | 内容过滤触发 |
| error | 服务器错误 |

#### GET /models

**请求格式**：无

**响应格式**：

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3-8b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "rustinfer"
    }
  ]
}
```

### 健康检查 API

#### GET /health

**响应格式**：

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "scheduler_connected": true,
  "tokenizer_loaded": true,
  "active_requests": 42
}
```

### 监控指标 API

#### GET /metrics

**响应格式** (Prometheus 格式)：

```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{endpoint="/v1/chat/completions",method="POST",status="200"} 1234

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/v1/chat/completions",le="0.1"} 100
http_request_duration_seconds_bucket{endpoint="/v1/chat/completions",le="0.5"} 500
http_request_duration_seconds_bucket{endpoint="/v1/chat/completions",le="+Inf"} 1000

# HELP active_requests_current Number of active inference requests
# TYPE active_requests_current gauge
active_requests_current 42

# HELP cpu_usage_percent CPU usage percentage
# TYPE cpu_usage_percent gauge
cpu_usage_percent 75.5

# HELP memory_usage_bytes Memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes 8589934592
```

---

## 通信协议

### Server ↔ Scheduler 协议

#### 传输层

- **协议**：ZeroMQ DEALER/PATTERN
- **序列化**：MessagePack (msgpack)
- **地址**：`tcp://scheduler-host:5555`

#### Server → Scheduler 命令

**SchedulerCommand** (MessagePack 编码):

```rust
pub enum SchedulerCommand {
    AddRequest(InferRequest),    // 添加推理请求
    AbortRequest(String),        // 取消请求 (request_id)
}
```

**InferRequest** 结构：

```rust
pub struct InferRequest {
    pub request_id: String,        // 请求唯一标识
    pub tokens: Vec<u32>,          // Token IDs
    pub sampling_params: SamplingParams,  // 采样参数
}
```

**SamplingParams** 结构：

```rust
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}
```

#### Scheduler → Server 响应

**SchedulerOutput** (MessagePack 编码):

```rust
pub enum SchedulerOutput {
    Step(StepOutput),      // 中间生成步骤
    Finish(FinishOutput),  // 请求完成
    Error(ErrorOutput),    // 错误信息
}
```

**StepOutput** 结构（轻量级）：

```rust
pub struct StepOutput {
    pub request_id: String,    // 请求 ID
    pub token: u32,            // 生成的 Token ID
}
```

**FinishOutput** 结构：

```rust
pub struct FinishOutput {
    pub request_id: String,
    pub finish_reason: FinishReason,  // 完成原因
    pub total_tokens: u32,             // 总 Token 数
}
```

**ErrorOutput** 结构：

```rust
pub struct ErrorOutput {
    pub request_id: String,
    pub error: String,         // 错误消息
    pub code: u16,              // 错误代码
}
```

#### 消息流程

```
Client Request
    │
    ▼
HTTP Handler
    │
    ├─▶ 生成 request_id
    │
    ├─▶ Tokenize: [Text] → [Token IDs]
    │
    ▼
SchedulerCommand::AddRequest(InferRequest)
    │
    ▼ (MessagePack serialize)
ZeroMQ Send
    │
    ─────────────────────────────▶ Scheduler
                                      │
                                      ├─▶ Scheduler (schedule)
                                      │
                                      ├─▶ Worker (inference)
                                      │
                                      └─▶ Worker (generate)
                                              │
                                              ▼ (Token ID)
                                     SchedulerOutput::Step
                                              │
                                              ▼ (MessagePack serialize)
                                     ZeroMQ Receive
                                              │
                                              ▼
                                     HTTP Handler
                                              │
                                              ├─▶ Detokenize: [Token ID] → [Text]
                                              │
                                              ▼
                                     SSE Event: "data: {...}\n\n"
                                              │
                                              ▼
                                     Client
```

### 消息大小优化

| 阶段 | 数据格式 | 大小估算 |
|------|----------|----------|
| Client → Server | UTF-8 文本 | ~1KB |
| Server → Scheduler | Token IDs (u32) | ~4KB (1024 tokens) |
| Scheduler → Worker | Token IDs | ~4KB |
| Worker → Scheduler | Token ID | 4 bytes |
| Scheduler → Server | Token ID | 4 bytes |
| Server → Client | UTF-8 文本 | ~1KB |

**结论**：通过仅传输 Token IDs，将 Server-Scheduler 之间的数据传输量减少 75% 以上。

---

## 状态管理

### 全局状态 (AppState)

```rust
pub struct AppState {
    // 配置（不可变，共享）
    pub config: Arc<ServerConfig>,

    // Tokenizer（不可变，共享）
    pub tokenizer: Arc<TokenizerWrapper>,

    // Scheduler 通信通道（单向）
    pub scheduler_tx: mpsc::UnboundedSender<SchedulerCommand>,

    // 请求映射表（读写锁）
    pub request_map: Arc<RwLock<RequestMap>>,
}
```

### 请求生命周期

#### 1. 请求注册

```rust
pub fn register_request(
    &self,
    request_id: String,
    tx: mpsc::UnboundedSender<SchedulerOutput>,
) -> Result<()> {
    let mut map = self.request_map.write().await;
    map.insert(request_id, tx);
    Ok(())
}
```

#### 2. 请求路由

```rust
pub async fn route_output(&self, output: SchedulerOutput) {
    let request_id = output.request_id();
    let map = self.request_map.read().await;

    if let Some(tx) = map.get(&request_id) {
        let _ = tx.send(output);
    }
}
```

#### 3. 请求清理

```rust
pub fn unregister_request(&self, request_id: &str) {
    let mut map = self.request_map.write().await;
    map.remove(request_id);
}
```

### 线程安全保证

| 资源 | 保护方式 | 并发访问模式 |
|------|----------|--------------|
| config | Arc | 读多写少（不可变） |
| tokenizer | Arc | 读多写少（不可变） |
| scheduler_tx | 内部 Mutex | 写多（单生产者） |
| request_map | Arc<RwLock> | 读多写少 |

### 性能优化

1. **Tokenizer 共享**：避免每个请求重复加载 Tokenizer
2. **零拷贝**：使用 `Arc` 共享不可变数据，无需复制
3. **无锁通道**：使用 `mpsc::unbounded_channel` 避免锁竞争
4. **读写锁**：`request_map` 使用读写锁，允许多个请求同时路由

---

## 扩展指南

### 添加新的采样参数

**步骤**：

1. 修改 `http/request.rs` 中的 `ChatCompletionRequest`
2. 修改 `api/openai.rs` 中的 `to_sampling_params()`
3. 更新 `infer-protocol` 中的 `SamplingParams` 结构

**示例**：

```rust
// http/request.rs
pub struct ChatCompletionRequest {
    // ... existing fields
    pub repetition_penalty: Option<f32>,  // 新增
}

// api/openai.rs
impl ChatCompletionRequest {
    pub fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams {
            // ... existing fields
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
        }
    }
}
```

### 添加新的 API 端点

**步骤**：

1. 在 `api/` 目录下创建新文件
2. 实现处理器函数
3. 在 `main.rs` 中注册路由

**示例**：

```rust
// api/embeddings.rs
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    req: Json<EmbeddingRequest>,
) -> Result<Response> {
    // 实现
}

// main.rs
let app = Router::new()
    .route("/v1/embeddings", post(api::embeddings::embeddings))
    // ...
```

### 添加新的对话模板

**步骤**：

1. 在 `chat/template.rs` 中定义新的模板结构
2. 实现 `ChatTemplate` trait
3. 在 `config.rs` 中添加模板配置

**示例**：

```rust
// chat/template.rs
pub trait ChatTemplate: Send + Sync {
    fn apply(&self, messages: &[Message]) -> Result<String>;
}

pub struct Llama3Template;
impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[Message]) -> Result<String> {
        // LLaMA 3 模板实现
    }
}

pub struct QwenTemplate;
impl ChatTemplate for QwenTemplate {
    fn apply(&self, messages: &[Message]) -> Result<String> {
        // Qwen 模板实现
    }
}
```

### 添加自定义监控指标

**步骤**：

1. 在 `api/metrics.rs` 中定义新指标
2. 在相应位置更新指标
3. 确保线程安全

**示例**：

```rust
// api/metrics.rs
use prometheus::{IntCounter, IntGauge};

lazy_static! {
    static ref CUSTOM_METRIC: IntCounter = register_int_counter!(
        "custom_metric_total",
        "Total count of custom metric"
    ).unwrap();
}

// 更新指标
CUSTOM_METRIC.inc();
```

### 支持多 LoRA

**步骤**：

1. 在 `http/request.rs` 中添加 LoRA 适配器字段
2. 在 `InferRequest` 中传递 LoRA 信息
3. 更新 Scheduler 协议支持 LoRA 路由

**示例**：

```rust
// http/request.rs
pub struct ChatCompletionRequest {
    // ... existing fields
    pub lora_adapter: Option<String>,  // LoRA 适配器名称
}

// 构造 InferRequest
let command = SchedulerCommand::AddRequest(InferRequest {
    // ... existing fields
    lora_adapter: req.lora_adapter,
});
```

---

## 最佳实践

### 错误处理

```rust
// 统一错误类型
pub type Result<T> = std::result::Result<T, ServerError>;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] TokenizerError),
}

// HTTP 响应转换
impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ServerError::SchedulerError(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
            ServerError::TokenizerError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.to_string()),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}
```

### 日志记录

```rust
use tracing::{info, warn, error};

// 结构化日志
info!(
    request_id = %request_id,
    model = %model_name,
    tokens = tokens.len(),
    "Processing inference request"
);

// 错误日志
error!(
    error = %err,
    request_id = %request_id,
    "Failed to route scheduler output"
);
```

### 测试

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chat_completions() {
        // 测试聊天完成接口
    }

    #[tokio::test]
    async fn test_stream_decoder() {
        // 测试流式解码器
    }
}
```

---

## 性能基准

### 目标指标

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| 首字延迟 (TTFT) | < 50ms | TBD |
| Token 生成延迟 | < 10ms/token | TBD |
| 并发请求 | > 1000 | TBD |
| HTTP QPS | > 10,000 | TBD |
| 内存占用 | < 2GB | TBD |

### 优化建议

1. **减少序列化开销**：考虑使用更高效的序列化格式 (如 bincode)
2. **连接池**：对于未来可能的多 Scheduler 部署，使用连接池
3. **压缩**：对于大 Prompt，考虑传输前压缩
4. **缓存**：缓存常见对话模板的 Tokenize 结果
5. **异步优化**：使用 `tokio` 运行时优化任务调度

---

## 参考资源

- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [ZeroMQ 文档](https://zeromq.org/get-started/)
- [MessagePack 规范](https://msgpack.org/)
- [Axum 文档](https://docs.rs/axum/)
- [Prometheus 指标规范](https://prometheus.io/docs/practices/naming/)
