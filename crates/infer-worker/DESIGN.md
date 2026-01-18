# Infer-Worker 设计文档

## 1. 设计哲学

### 1.1 核心原则

**单一职责（Single Responsibility）**
Worker 只负责模型推理计算，不参与调度决策、请求管理或业务逻辑。它是一个纯粹的计算单元，专注于在指定设备（CPU/GPU）上高效执行模型前向传播。

**关注点分离（Separation of Concerns）**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Server    │ -> │  Scheduler  │ -> │   Worker    │
│   (API)     │    │ (Orchest-   │    │ (Compute)   │
│             │    │   ration)   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```
- **Server**: HTTP 接口、文本处理、流式输出
- **Scheduler**: 请求调度、内存管理、批处理构建
- **Worker**: 模型加载、张量计算、KV Cache 操作

**设备抽象（Device Abstraction）**
通过 `DeviceType` 和统一的数据传输 API，实现 CPU/CUDA 的无缝切换。上层代码无需关心底层硬件，通过自动的 `to_device()` 转换实现最优性能。

**零拷贝原则（Zero-Copy）**
在张量操作、权重加载、内存传输等关键路径上，通过引用、切片、内存映射等技术避免不必要的数据拷贝，最大化推理性能。

### 1.2 性能导向

**最小化序列化开销**
使用 `bincode` 进行紧凑的二进制序列化，避免文本协议的冗余。通信消息只传输必要的元数据，张量数据通过共享内存或直接指针传递（未来）。

**异步非阻塞**
ZeroMQ 的异步 I/O 模型允许 Worker 同时处理多个请求。虽然推理本身是同步的，但通信层不阻塞其他 Worker 的调度。

**内存预分配**
KV Cache 和计算缓冲区在初始化时预先分配，避免推理过程中的动态分配导致的性能抖动。

### 1.3 可扩展性

**插件化模型系统**
通过 `Model` trait 和 `ModelFactory` 实现运行时模型注册，支持 Llama、Qwen、Mistral 等多种架构，无需修改 Worker 核心代码。

**类型安全张量**
通过 `Tensor<T>` 枚举实现类型安全的动态张量系统，支持 F32、BF16、I8 等多种数据类型，编译期保证类型正确性。

## 2. 服务边界

### 2.1 职责范围

#### ✅ Worker 负责

| 功能 | 描述 |
|------|------|
| **模型管理** | 加载、卸载模型，维护模型实例生命周期 |
| **KV Cache** | 管理 KV Cache 内存池，提供页式访问接口 |
| **张量计算** | 执行矩阵乘法、激活函数、层归一化等操作 |
| **采样** | 实现温度、Top-P、Top-K 等采样策略 |
| **设备管理** | 管理 CUDA Stream、cuBLAS 句柄等设备资源 |
| **状态监控** | 收集内存使用、计算延迟等统计信息 |
| **健康检查** | 响应心跳和状态查询请求 |

#### ❌ Worker 不负责

| 功能 | 负责方 |
|------|--------|
| **请求调度** | Scheduler |
| **批处理构建** | Scheduler |
| **内存分配策略** | Scheduler（只提供接口）|
| **文本处理** | Server |
| **流式输出** | Server |
| **负载均衡** | Scheduler |

### 2.2 外部依赖

```
┌─────────────────────────────────────────────────────┐
│                    Worker                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Model Layer  │    │   Tensor Operations      │  │
│  │ (Llama3/Qwen)│    │   (matmul/softmax/etc)   │  │
│  └──────┬───────┘    └──────────────────────────┘  │
│         │                      ↑                    │
│  ┌──────▼───────┐    ┌────────┴─────────────────┐   │
│  │  Base Types  │    │      CUDA / CPU          │   │
│  │ (Device/     │────│      Device Layer        │   │
│  │  Allocator)  │    │  (cuBLAS/cuRAND/etc)    │   │
│  └──────────────┘    └───────────────────────────┘   │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Communication: ZeroMQ (bincode)                     │
│  ↓                                                   │
│  Scheduler                                           │
└─────────────────────────────────────────────────────┘
```

### 2.3 依赖隔离

- **无网络库依赖**: 使用 ZeroMQ 处理所有网络通信，无需 HTTP、gRPC 等协议
- **无日志框架硬编码**: 使用 `log` trait，允许上层配置 `env_logger`、`tracing` 等
- **无配置文件**: 配置通过环境变量或启动参数传递，简化部署

## 3. 架构设计

### 3.1 模块结构

```
crates/infer-worker/src/
├── lib.rs                    # 公共导出
├── bin/worker_main.rs        # 可执行入口
│
├── worker/                   # Worker 核心层
│   ├── worker.rs            # Worker 结构体和核心逻辑
│   ├── server.rs            # ZeroMQ RPC Server
│   ├── config.rs            # WorkerConfig 配置
│   └── device_info.rs       # DeviceInfo 设备信息
│
├── model/                    # 模型抽象层
│   ├── mod.rs               # Model trait 定义
│   ├── llama3.rs           # Llama3 实现
│   ├── config.rs           # ModelConfig 模型配置
│   ├── kvcache.rs          # KVCache 管理
│   ├── safetensor_loader.rs # 权重加载器
│   ├── tokenizer.rs        # 分词器（仅用于验证）
│   ├── factory.rs          # ModelFactory 工厂
│   ├── registry.rs         # ModelRegistry 注册表
│   └── layers/             # 通用层抽象
│
├── tensor/                   # 张量层
│   ├── mod.rs               # Tensor 枚举和操作
│   └── ops.rs               # 运算实现
│
├── cuda/                     # CUDA 支持层
│   ├── mod.rs               # 模块导出
│   ├── device.rs            # CUDA 设备管理
│   ├── config.rs            # CUDA 配置
│   ├── ffi.rs               # CUDA FFI 绑定
│   └── error.rs             # CUDA 错误处理
│
├── base/                     # 基础设施层
│   ├── mod.rs               # DeviceType/DataType
│   ├── error.rs             # Error 类型
│   ├── allocator.rs         # 内存分配器
│   └── buffer.rs            # Buffer 抽象
│
└── op/                       # 算子层（未来扩展）
    └── kernels/             # CUDA Kernel
```

### 3.2 核心组件

#### Worker 结构

```rust
pub struct Worker {
    // 基础配置
    config: WorkerConfig,
    device_type: DeviceType,

    // 模型相关
    model: Option<Box<dyn Model>>,

    // 内存管理
    kv_cache_pool: Option<KVCachePool>,
    buffer_allocator: Arc<dyn Allocator>,

    // 设备资源
    cuda_stream: Option<CudaStream>,
    cublas_handle: Option<CuBLASHandle>,

    // 统计信息
    stats: WorkerStats,
}
```

**生命周期管理**:
1. **创建**: `Worker::new(config)` - 初始化设备资源
2. **注册**: 连接到 Scheduler，发送设备信息
3. **加载**: `load_model()` - 加载模型权重
4. **初始化**: `init_kv_cache()` - 预分配 KV Cache
5. **推理**: `forward()` / `forward_with_cache()` - 执行计算
6. **销毁**: 清理 CUDA 资源，释放内存

#### WorkerServer 结构

```rust
pub struct WorkerServer {
    worker: Arc<Worker>,
    zmq_socket: zmq::Socket,
    scheduler_addr: String,
}
```

**通信模式**: ZeroMQ Dealer-Router 模式
```
┌──────────────┐          ┌──────────────┐
│   Scheduler  │◄────────►│    Worker     │
│   (Router)   │          │   (Dealer)    │
└──────────────┘          └──────────────┘
       │                         │
       └───── 多个 Worker ─────────┘
```

### 3.3 数据流

#### 前向传播流程

```
1. Scheduler 发送请求
   WorkerCommand::Forward(ForwardParams {
       input_ids: Vec<u32>,
       positions: Vec<usize>,
       cache_slots: Option<Vec<(usize, usize)>>,
   })

2. Worker 解析参数
   └─> input_ids → Tensor<u32> [batch, seq_len]
   └─> positions → Tensor<i64> [batch, seq_len]
   └─> cache_slots → 更新 KV Cache 索引

3. 执行模型
   input → Embedding → Blocks (Attention + FFN) → LM Head → logits

4. 采样（如需要）
   logits → temperature → top_p/top_k → sample → token_id

5. 返回结果
   WorkerResponse::ForwardCompleted(ForwardResult {
       output_logits: Vec<f32>,
       output_tokens: Vec<u32>,
       compute_time_ms: u64,
   })
```

#### KV Cache 操作流程

```
Paged KV Cache 架构:

┌─────────────────────────────────────────────────┐
│              KV Cache Pool                      │
├─────────────────────────────────────────────────┤
│  Block 0  │ Block 1  │ Block 2  │ ... │ Block N │
│  (4096)   │  (4096)  │  (4096)  │     │  (4096) │
└─────────────────────────────────────────────────┘
     ▲           ▲           ▲
     │           │           │
  Seq A[0]   Seq A[1]   Seq B[0]
     │           │           │
  ┌──▼───────┐ ┌─▼────────┐ ┌─▼──────┐
  │ Sequence │ │ Sequence │ │Sequence│
  │    A     │ │    A     │ │   B    │
  └──────────┘ └──────────┘ └────────┘

每个 Block 存储固定长度的 KV 对，通过 block_id 和 offset 索引。
Copy-on-Write 支持序列间的前缀共享。
```

## 4. API 设计

### 4.1 Worker API（内部）

#### 创建和初始化

```rust
impl Worker {
    /// 创建新的 Worker 实例
    pub fn new(config: WorkerConfig) -> Result<Self>;

    /// 获取设备信息
    pub fn device_info(&self) -> DeviceInfo;
}
```

#### 模型管理

```rust
impl Worker {
    /// 加载模型
    pub fn load_model(&mut self, model: Box<dyn Model>) -> Result<()>;

    /// 卸载模型并释放内存
    pub fn unload_model(&mut self) -> Result<()>;

    /// 获取当前模型配置
    pub fn model_config(&self) -> Option<ModelConfig>;
}
```

#### KV Cache 管理

```rust
impl Worker {
    /// 初始化 KV Cache 池
    pub fn init_kv_cache(
        &mut self,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<KVCacheInfo>;

    /// 重置指定序列的 KV Cache
    pub fn reset_sequence_cache(&mut self, seq_id: usize) -> Result<()>;

    /// 复制 Cache（用于前缀共享）
    pub fn copy_cache(
        &mut self,
        src_seq: usize,
        dst_seq: usize,
        length: usize,
    ) -> Result<()>;
}
```

#### 推理接口

```rust
impl Worker {
    /// 简单前向传播（无 KV Cache）
    pub fn forward(
        &self,
        input: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor>;

    /// 带 KV Cache 的前向传播
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor>;

    /// Paged Attention 前向传播
    pub fn forward_paged(
        &self,
        input: &Tensor,
        positions: &Tensor,
        block_ids: &[Vec<usize>],
        offsets: &[Vec<usize>],
    ) -> Result<Tensor>;

    /// 采样
    pub fn sample(
        &self,
        logits: &Tensor,
        output: &mut Tensor,
        temperature: f32,
        top_p: f32,
        top_k: i32,
    ) -> Result<()>;
}
```

#### 状态和统计

```rust
impl Worker {
    /// 获取当前状态
    pub fn status(&self) -> WorkerStatus;

    /// 获取性能统计
    pub fn stats(&self) -> WorkerStats;

    /// 重置统计
    pub fn reset_stats(&mut self);
}
```

### 4.2 RPC API（Scheduler ↔ Worker）

#### WorkerCommand（Scheduler → Worker）

```rust
pub enum WorkerCommand {
    /// Worker 注册
    Register(WorkerRegistration),

    /// 加载模型
    LoadModel(ModelLoadParams),

    /// 初始化 KV Cache
    InitKVCache(InitKVCacheParams),

    /// 前向传播
    Forward(ForwardParams),

    /// Paged 前向传播
    ForwardPaged(PagedForwardParams),

    /// 获取状态
    GetStatus,

    /// 健康检查
    HealthCheck,

    /// 关闭 Worker
    Shutdown,
}
```

#### WorkerResponse（Worker → Scheduler）

```rust
pub enum WorkerResponse {
    /// 注册确认
    RegisterAck(WorkerRegistrationAck),

    /// 模型加载完成
    ModelLoaded(ModelLoadedInfo),

    /// KV Cache 初始化完成
    KVCacheInitialized(KVCacheInfo),

    /// 前向传播完成
    ForwardCompleted(ForwardResult),

    /// 错误响应
    Error(ErrorInfo),

    /// 状态信息
    Status(WorkerStatus),

    /// 健康检查响应
    HealthCheck(HealthCheckInfo),
}
```

### 4.3 Model API（可扩展）

```rust
pub trait Model: Send + Sync {
    /// 模型名称
    fn name(&self) -> &str;

    /// 模型配置
    fn config(&self) -> &ModelConfig;

    /// 简单前向传播
    fn forward(&self, input: &Tensor, pos: &Tensor) -> Result<Tensor>;

    /// 带 KV Cache 前向
    fn forward_with_cache(
        &self,
        input: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor>;

    /// Paged 前向传播
    fn forward_paged(
        &self,
        input: &Tensor,
        pos: &Tensor,
        block_ids: &[Vec<usize>],
        offsets: &[Vec<usize>],
    ) -> Result<Tensor>;

    /// 获取词表大小
    fn vocab_size(&self) -> usize;

    /// 获取隐藏层维度
    fn hidden_size(&self) -> usize;
}
```

## 5. 通信协议

### 5.1 消息格式

使用 `bincode` 进行二进制序列化：

```rust
// 消息头
struct MessageHeader {
    pub msg_type: u8,          // 消息类型
    pub payload_size: u32,     // 负载大小
    pub request_id: u64,       // 请求 ID（用于匹配响应）
}

// 完整消息
struct Message {
    pub header: MessageHeader,
    pub payload: Vec<u8>,      // bincode 序列化的 Command/Response
}
```

### 5.2 时序图

```
    Worker           Scheduler            Worker
      │                 │                   │
      │ ───────────────►│                   │
      │  Register       │                   │
      │                 │                   │
      │ ◄───────────────┤                   │
      │  RegisterAck    │                   │
      │                 │                   │
      │ ───────────────►│ ────────────────►  │
      │  LoadModel      │  LoadModel        │
      │                 │                   │
      │ ◄───────────────┤ ◄────────────────  │
      │  ModelLoaded    │  ModelLoaded      │
      │                 │                   │
      │ ───────────────►│ ────────────────►  │
      │  InitKVCache    │  InitKVCache      │
      │                 │                   │
      │ ◄───────────────┤ ◄────────────────  │
      │  CacheInit      │  CacheInit        │
      │                 │                   │
      │ ───────────────►│ ────────────────►  │
      │  ForwardPaged   │  ForwardPaged     │
      │                 │                   │
      │ ◄───────────────┤ ◄────────────────  │
      │  ForwardResult  │  ForwardResult    │
      │                 │                   │
```

### 5.3 错误处理

```rust
pub enum WorkerError {
    // 设备错误
    DeviceInitFailed(String),
    OutOfMemory { required: usize, available: usize },

    // 模型错误
    ModelLoadFailed(String),
    ModelNotFound,
    InvalidInputShape,

    // KV Cache 错误
    CacheNotInitialized,
    InvalidBlockId(u32),
    CacheFull,

    // 计算错误
    ComputeError(String),
    CudaError(CudaError),

    // 通信错误
    SerializationError,
    DeserializationError,
    Timeout,
}
```

## 6. 扩展性设计

### 6.1 新增模型支持

**步骤**:

1. 实现 `Model` trait:

```rust
impl Model for MyNewModel {
    fn name(&self) -> &str { "my-new-model" }
    fn config(&self) -> &ModelConfig { &self.config }
    fn forward(&self, input: &Tensor, pos: &Tensor) -> Result<Tensor> {
        // 实现前向传播
    }
    // ...
}
```

2. 注册到 ModelRegistry:

```rust
ModelRegistry::register("my-new-model", || {
    Box::new(MyNewModel::load(...)?)
});
```

3. 通过工厂创建:

```rust
let model = ModelFactory::create("my-new-model", path)?;
```

### 6.2 新增算子支持

**CPU 实现**:
```rust
pub fn my_cpu_op(input: &Tensor) -> Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(my_cpu_op_f32(t)?)),
        Tensor::BF16(t) => Ok(Tensor::BF16(my_cpu_op_bf16(t)?)),
        // ...
    }
}
```

**CUDA 实现**:
```rust
pub fn my_cuda_op(input: &Tensor, stream: &CudaStream) -> Result<Tensor> {
    let kernel = compile_cuda_kernel("my_op.cu")?;
    kernel.launch(input.as_ptr(), output.as_ptr(), size, stream)?;
    Ok(output)
}
```

**统一接口**:
```rust
impl Tensor {
    pub fn my_op(&self) -> Result<Tensor> {
        match self.device() {
            DeviceType::CPU => my_cpu_op(self),
            DeviceType::CUDA(id) => my_cuda_op(self, stream),
        }
    }
}
```

### 6.3 多 GPU 扩展

**张量并行（未来）**:

```rust
pub struct WorkerConfig {
    pub rank: usize,           // 本机 GPU 排名
    pub world_size: usize,     // 总 GPU 数量
    pub tensor_parallel_size: usize,
}

impl Worker {
    pub fn forward_distributed(
        &self,
        input: &Tensor,
        peers: &[PeerWorker],
    ) -> Result<Tensor> {
        // 1. AllReduce 聚合输入
        // 2. 分片计算
        // 3. AllGather 结果
    }
}
```

## 7. 性能优化

### 7.1 内存优化

| 技术 | 实现位置 | 效果 |
|------|---------|------|
| 零拷贝切片 | `Tensor::slice()` | 避免数据复制 |
| 内存映射权重 | `SafetensorLoader` | 减少加载时间 |
| 预分配缓冲区 | `KVCachePool` | 避免碎片化 |
| 统一内存 | CUDA `managed` | 简化数据传输 |

### 7.2 计算优化

| 技术 | 实现位置 | 效果 |
|------|---------|------|
| Flash Attention | `model/llama3.rs` | 减少 HBM 访问 |
| Kernel Fusion | `op/kernels/` | 减少启动开销 |
| 算子融合 | 前向传播 | 减少中间结果 |
| 异步执行 | CUDA Streams | 隐藏延迟 |

### 7.3 通信优化

| 技术 | 实现位置 | 效果 |
|------|---------|------|
| 二进制序列化 | bincode | 减少网络流量 |
| 消息批处理 | `BatchBuilder` | 减少往返次数 |
| ZeroMQ I/O 线程 | `zmq::Context` | 提高吞吐量 |

## 8. 监控和调试

### 8.1 统计信息

```rust
pub struct WorkerStats {
    // 计算统计
    pub total_requests: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,

    // 内存统计
    pub memory_allocated: usize,
    pub memory_reserved: usize,
    pub cache_hit_rate: f64,

    // 设备利用率
    pub gpu_utilization: f32,
    pub memory_bandwidth_util: f32,
}
```

### 8.2 日志级别

- `ERROR`: 不可恢复的错误（OOM、CUDA 错误）
- `WARN`: 可恢复的错误（超时、重试）
- `INFO`: 关键生命周期事件（模型加载、Cache 初始化）
- `DEBUG`: 请求处理细节
- `TRACE`: 张量形状、详细时间线

### 8.3 调试模式

```rust
pub struct WorkerConfig {
    pub debug_mode: bool,
    pub profile_kernels: bool,
    pub dump_tensors: bool,
    pub verify_results: bool,
}
```

## 9. 部署考虑

### 9.1 资源要求

| 组件 | 最小配置 | 推荐配置 |
|------|---------|---------|
| 内存 | 32GB | 64GB+ |
| GPU VRAM | 24GB | 48GB+ (A100/H100) |
| GPU Compute | 7.0+ | 8.0+ |
| CPU | 8核 | 16核+ |
| 网络 | 10Gbps | 25Gbps+ |

### 9.2 环境变量

```bash
# Worker 配置
WORKER_ID=worker-0
SCHEDULER_ADDR=tcp://scheduler:5555
DEVICE_TYPE=cuda
CUDA_DEVICE_ID=0

# 性能调优
ZMQ_IO_THREADS=4
ZMQ_MAX_SOCKETS=1024
RUST_LOG=info
```

### 9.3 容器化

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev

WORKDIR /app
COPY target/release/worker ./worker

CMD ["./worker"]
```

## 10. 未来规划

### 10.1 短期目标

- [ ] 实现完整的 Flash Attention 2
- [ ] 支持 BF16 计算和存储
- [ ] 添加更多模型（Qwen、Mistral）
- [ ] 完善性能分析和 Profiling

### 10.2 中期目标

- [ ] 实现张量并行
- [ ] 支持 LoRA Adapter
- [ ] 添加投机采样
- [ ] 实现 CUDA Graph 优化

### 10.3 长期目标

- [ ] 支持 INT8/INT4 量化
- [ ] 实现流水线并行
- [ ] 多模态模型支持
- [ ] 分布式训练支持

---

## 附录

### A. 参考资料

- [vLLM Architecture](https://docs.vllm.ai/en/latest/architecture/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)

### B. 相关代码

- `src/worker/worker.rs` - Worker 核心实现
- `src/model/mod.rs` - Model trait 定义
- `src/tensor/mod.rs` - 张量抽象
- `src/cuda/mod.rs` - CUDA 绑定

### C. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.0 | 2024-01 | 初始版本，基础推理功能 |
| 0.2.0 | 2024-02 | 添加 Paged KV Cache |
| 0.3.0 | 2024-03 | 支持多模型注册 |
