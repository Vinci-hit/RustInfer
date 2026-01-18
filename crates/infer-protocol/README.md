# infer-protocol

RustInfer 项目的通信协议定义库。提供 Server、Engine 和 Frontend 之间的通用数据结构和序列化格式。

## 概述

`infer-protocol` 是一个共享的协议库，定义了以下组件之间的通信格式：

- **infer-server** (REST API 服务器)
- **infer-engine** (推理引擎进程)
- **infer-frontend** (Web 前端)

所有数据结构都使用 `serde` 进行序列化/反序列化，支持 JSON 和 MessagePack 等多种格式。

## 数据结构

### 1. 引擎和缓存指标

#### `EngineMetrics` - 引擎和缓存完整指标
```rust
pub struct EngineMetrics {
    pub engine: EngineStats,  // 引擎统计
    pub cache: CacheStats,     // 缓存统计
}
```

#### `EngineStats` - 引擎统计指标
```rust
pub struct EngineStats {
    pub total_requests: u64,           // 总请求数
    pub completed_requests: u64,       // 完成的请求数
    pub failed_requests: u64,          // 失败的请求数
    pub total_tokens_generated: u64,   // 生成的总token数
    pub avg_queue_time_ms: f64,        // 平均排队时间
    pub avg_prefill_time_ms: f64,      // 平均prefill时间
    pub avg_decode_time_ms: f64,        // 平均decode时间
    pub queue_size: usize,              // 当前队列大小
    pub queue_capacity: usize,          // 队列容量
    pub concurrent_requests: usize,     // 当前并发请求数
}
```

#### `CacheStats` - 缓存统计指标
```rust
pub struct CacheStats {
    pub hit_rate: f64,         // 缓存命中率 (0.0 - 1.0)
    pub hits: u64,             // 命中次数
    pub misses: u64,           // 未命中次数
    pub evictable_size: usize, // 可驱逐的缓存大小 (tokens)
    pub protected_size: usize, // 受保护的缓存大小 (tokens)
    pub total_cached: usize,   // 总缓存大小 (tokens)
    pub total_capacity: usize, // 总缓存容量 (tokens)
    pub evictions: u64,        // 驱逐次数
    pub node_count: usize,     // 节点数量
}
```

### 2. ZMQ 协议消息

#### `EngineRequest` - Server → Engine 请求
```rust
pub enum EngineRequest {
    Inference(InferenceRequest),  // 推理请求
    MetricsQuery,                // 查询指标
}
```

#### `EngineResponse` - Engine → Server 响应
```rust
pub enum EngineResponse {
    Inference(InferenceResponse),  // 推理响应
    Metrics(EngineMetrics),        // 指标响应
}
```

### 3. 推理消息类型

#### `InferenceRequest` - 推理请求
```rust
pub struct InferenceRequest {
    pub request_id: String,          // 唯一请求ID (UUID v4)
    pub prompt: String,              // 输入文本 (已应用chat template)
    pub max_tokens: usize,           // 最大生成tokens数量
    pub temperature: f32,            // 温度参数 (0.0 = greedy, 1.0 = random)
    pub top_p: f32,                  // Top-p (nucleus) sampling
    pub top_k: i32,                  // Top-k sampling
    pub stop_sequences: Vec<String>, // 停止序列
    pub stream: bool,                // 是否流式返回
    pub priority: i32,               // 优先级 (0=normal, 1=high, -1=low)
}
```

#### `InferenceResponse` - 推理响应
```rust
pub struct InferenceResponse {
    pub request_id: String,              // 对应的请求ID
    pub status: ResponseStatus,          // 响应状态
    pub text: Option<String>,            // 生成的文本 (非流式时返回)
    pub tokens: Option<Vec<i32>>,        // Token IDs (可选，调试用)
    pub num_tokens: u32,                 // 生成的token数量
    pub error: Option<String>,           // 错误信息
    pub metrics: InferenceMetrics,       // 性能指标
}
```

#### `InferenceMetrics` - 性能指标
```rust
pub struct InferenceMetrics {
    pub prefill_ms: u64,          // Prefill阶段耗时 (ms)
    pub decode_ms: u64,            // Decode阶段耗时 (ms)
    pub queue_ms: u64,             // 排队等待时间 (ms)
    pub batch_size: usize,         // 实际batch大小
    pub tokens_per_second: f64,    // 吞吐量 (tokens/s)
    pub decode_iterations: usize,   // Decode迭代次数
}
```

### 4. 流式消息

#### `StreamChunk` - 流式响应chunk
```rust
pub struct StreamChunk {
    pub request_id: String,                 // 请求ID
    pub chunk_type: ChunkType,              // chunk类型
    pub token: Option<String>,              // token文本
    pub token_id: Option<i32>,             // token ID
    pub finish_reason: Option<String>,       // 完成原因
    pub metrics: Option<InferenceMetrics>,  // 性能指标
}
```

#### `ChunkType` - Chunk类型
```rust
pub enum ChunkType {
    Token,  // 正常token
    Done,   // 生成完成
    Error,  // 错误
}
```

## 使用示例

### 在 Server 中使用

```rust
use infer_protocol::{EngineRequest, EngineResponse, InferenceRequest};

// 创建推理请求
let request = EngineRequest::Inference(InferenceRequest {
    request_id: uuid::Uuid::new_v4().to_string(),
    prompt: "Hello, world!".to_string(),
    max_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50,
    stop_sequences: vec![],
    stream: false,
    priority: 0,
});

// 通过 ZMQ 发送到 Engine
zmq_socket.send(&serde_json::to_vec(&request)?)?;
```

### 在 Engine 中使用

```rust
use infer_protocol::{EngineResponse, InferenceResponse, ResponseStatus};

// 创建推理响应
let response = EngineResponse::Inference(InferenceResponse {
    request_id: request_id.clone(),
    status: ResponseStatus::Success,
    text: Some("Hello! How can I help you?".to_string()),
    tokens: Some(vec![12345, 67890]),
    num_tokens: 7,
    error: None,
    metrics: InferenceMetrics {
        prefill_ms: 45,
        decode_ms: 123,
        queue_ms: 0,
        batch_size: 1,
        tokens_per_second: 14.5,
        decode_iterations: 7,
    },
});

// 发送回 Server
zmq_socket.send(&serde_json::to_vec(&response)?)?;
```

### 在 Frontend 中使用

```rust
use infer_protocol::{EngineStats, CacheStats};

// 解析响应
let metrics: EngineMetrics = serde_json::from_str(json_string)?;

// 访问指标
println!("Total requests: {}", metrics.engine.total_requests);
println!("Cache hit rate: {:.2}%", metrics.cache.hit_rate * 100.0);
```

## 依赖

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

## 依赖此 crate 的组件

- `infer-server` - REST API 服务器
- `infer-engine` - 推理引擎
- `infer-frontend` - Web 前端 (通过 API 间接使用)

## 架构说明

```
┌─────────────┐         HTTP/REST          ┌──────────────┐
│  Frontend   │ ◄─────────────────────────► │    Server    │
│  (Web UI)   │                            │ (HTTP API)   │
└─────────────┘                            └──────┬───────┘
                                                  │
                                            ZMQ/JSON
                                                  │
                                           ┌──────▼──────┐
                                           │    Engine    │
                                           │  (Inference) │
                                           └─────────────┘
```

1. **Frontend → Server**: HTTP REST API (JSON)
2. **Server → Engine**: ZMQ (JSON 序列化的协议消息)
3. **Engine → Server**: ZMQ (JSON 序列化的协议消息)
4. **Server → Frontend**: HTTP REST API (JSON)

## 协议版本

当前版本: `1.0`

协议变更规则:
- 新增字段: 向后兼容
- 删除字段: 需要主版本号升级
- 重命名字段: 需要主版本号升级

## 测试

运行测试:
```bash
cargo test -p infer-protocol
```
