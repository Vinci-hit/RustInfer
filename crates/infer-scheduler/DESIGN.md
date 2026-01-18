# RustInfer Scheduler 设计文档

> 版本: v0.1.0
> 更新日期: 2026-01-18

本文档是 infer-scheduler 的"单一事实来源"（Single Source of Truth），帮助快速理解调度器架构而无需扫描整个代码库。

---

## 目录

- [设计哲学](#设计哲学)
- [服务边界](#服务边界)
- [架构概览](#架构概览)
- [核心模块](#核心模块)
- [通信协议](#通信协议)
- [内存管理](#内存管理)
- [调度策略](#调度策略)
- [状态管理](#状态管理)
- [配置系统](#配置系统)
- [API 参考](#api-参考)
- [扩展指南](#扩展指南)
- [最佳实践](#最佳实践)

---

## 设计哲学

### 1. 纯 Token 层架构

**核心原则**: Scheduler 只处理 Token IDs，不涉及任何文本处理。

**设计决策**:
- Tokenizer 位于 Server 层，Scheduler 层不持有 Tokenizer
- Server ↔ Scheduler 传输的是 Token IDs (`Vec<u32>`)
- 计算边界明确，避免重复的字符串序列化/反序列化

**权衡**:
- ✅ 优势: 减少 50%+ 的网络传输开销，降低序列化延迟

```rust
// ✅ 正确: Scheduler 只处理 Token IDs
pub struct InferRequest {
    pub request_id: String,
    pub input_tokens: Vec<i32>,  // Token IDs，不是文本
    // ...
}

// ❌ 错误: 不要在 Scheduler 中持有 Tokenizer
```

---

### 2. PagedAttention 内存管理

**核心原则**: 类似 vLLM 的分页 KV Cache，避免内存碎片。

**关键特性**:
- 物理块（Physical Block）和逻辑块（Logical Block）分离
- 动态分配/释放，支持序列长度扩展
- 引用计数机制，支持 CoW（Copy-on-Write）

**数据结构**:
```rust
// 物理块分配器
pub struct BlockAllocator {
    free_list: Vec<PhysicalBlockId>,    // 空闲块链表
    block_refs: HashMap<PhysicalBlockId, usize>,  // 引用计数
    total_blocks: usize,
}

// 序列块表
pub struct BlockTable {
    blocks: Vec<PhysicalBlockId>,       // 逻辑块 → 物理块映射
    ref_counts: Vec<usize>,            // 每个块的引用计数
}
```

**权衡**:
- ✅ 优势: 高内存利用率，支持动态序列长度，避免碎片
- ❌ 劣势: 实现复杂，需要维护块映射关系

---

### 3. 前缀缓存（RadixTree）

**核心原则**: 复用相同前缀的 KV Cache，减少重复计算。

**应用场景**:
- System Prompt 复用（所有请求共享相同的前缀）
- 对话历史重用（多轮对话）
- 代码生成（重复的代码片段）

**算法**:
```rust
pub struct RadixTree {
    root: RadixNode,
}

impl RadixTree {
    // 查找最长匹配的前缀
    pub fn match_prefix(&mut self, tokens: &[i32]) -> Option<PrefixMatch> {
        // 返回: (匹配的 tokens, 对应的物理块列表, 引用计数)
    }

    // 插入新的前缀
    pub fn insert(&mut self, tokens: &[i32], blocks: &[PhysicalBlockId]) {
        // 支持节点分裂，部分复用
    }
}
```

**性能提升**:
- 对于相同的 System Prompt，可以节省 90%+ 的 Prefill 时间
- 对话场景下，缓存命中率通常可达 70-80%

---

### 4. 五阶段调度循环

**核心原则**: 将调度逻辑分解为清晰的五个阶段，职责单一。

**调度循环** (`coordinator.rs`):
```rust
async fn run(&mut self) {
    loop {
        // Phase 1: Ingest - 接收新请求
        self.ingest_requests().await;

        // Phase 2: Schedule - 调用策略生成批次计划
        let plan = self.schedule_requests();

        // Phase 3: Apply - 应用计划（抢占、分配内存、更新状态）
        self.apply_plan(&plan).await?;

        // Phase 4: Execute - 调用 Worker 执行推理
        self.execute_batch(&plan).await?;

        // Phase 5: Post-Process - 处理输出，更新缓存
        self.post_process(&plan).await;

        // Idle Sleep
        tokio::time::sleep(Duration::from_millis(self.config.idle_sleep_ms)).await;
    }
}
```

**权衡**:
- ✅ 优势: 逻辑清晰，易于调试，每个阶段可独立优化
- ❌ 劣势: 增加复杂度，需要维护跨阶段的状态一致性

---

### 5. 流式处理优先

**核心原则**: 优先支持流式输出，降低首字延迟（TTFT）。

**实现**:
```rust
// 生成单个 token 后立即返回
pub struct StepOutput {
    pub request_id: String,
    pub token_id: u32,
    pub position: usize,
}

// 通过 channel 立即推送给 Server
if let Some(tx) = sequence.output_tx.as_ref() {
    let _ = tx.send(step_output);
}
```

**性能目标**:
- TTFT (Time To First Token) < 100ms
- Token Throughput > 50 tokens/s

---

### 6. 可插拔调度策略

**核心原则**: 调度策略抽象化，支持多种策略。

**接口定义**:
```rust
pub trait SchedulingPolicy: Send + Sync {
    /// 根据当前状态生成批次计划
    fn schedule(&self, context: &ScheduleContext) -> BatchPlan;

    /// 策略名称
    fn name(&self) -> &'static str;

    /// 策略配置
    fn config(&self) -> &SchedulePolicyConfig;
}
```

**默认策略**: `ContinuousBatchingPolicy`（连续批处理）
- 优先处理 Decode 任务（已有 KV Cache）
- 充分利用 GPU，提高 Throughput
- 支持抢占（Preemption）

---

## 服务边界

### Scheduler 负责什么？

| 职责 | 说明 |
|-----|------|
| **请求调度** | 管理请求队列，决定何时、何人执行 |
| **内存管理** | 分配/释放 KV Cache 物理块，维护 Block Table |
| **前缀缓存** | 使用 RadixTree 复用 KV Cache |
| **Worker 协调** | 管理多个 Worker 生命周期，调用推理接口 |
| **批处理** | 将多个请求合并为批次，提高 GPU 利用率 |
| **抢占** | 低优先级请求腾出资源给高优先级请求 |
| **流式输出** | 接收 Worker 结果，立即转发给 Server |

### Scheduler 不负责什么？

| 职责 | 为什么不负责 |
|-----|------------|
| **Tokenizer** | Tokenizer 在 Server 层，Scheduler 只处理 Token IDs |
| **模型推理** | 推理由 Worker 执行，Scheduler 只做调度 |
| **HTTP 服务** | HTTP 由 Server 层处理 |
| **用户认证** | 认证/授权应在网关层完成 |
| **文本后处理** | 文本解码、格式化由 Server 负责 |

### 与其他组件的边界

```
┌─────────────────────────────────────────────────────────────┐
│                       Server 层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  HTTP Server │  │   Tokenizer  │  │  Text Utils  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                        │ Token IDs (Vec<u32>)
                        │ ZMQ PUSH/PULL
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Scheduler 层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Coordinator  │  │MemoryManager │  │ RadixTree    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                        │ ForwardParams
                        │ ZMQ ROUTER/DEALER
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Worker 层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   LLaMA3    │  │  KV Cache    │  │  CUDA Graph  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 架构概览

### 三层架构

```text
┌──────────────────────────────────────────────────────────────────┐
│                         HTTP Clients                         │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTP POST /v1/chat/completions
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Server Layer                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │   HTTP     │  │ Tokenizer  │  │  Stream    │    │   │
│  │  │  Server    │  │            │  │  Decoder   │    │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │   │
│  │        │              │              │              │    │   │
│  └────────┼──────────────┼──────────────┼──────────────┘    │
│           │ Token IDs    │              │ SSE Stream          │
└───────────┼──────────────┼──────────────┼───────────────────┘
            │              │              │
            │              │              ▼
            │              │      ┌──────────────┐
            │              │      │ HTTP Client  │
            │              │      │ (浏览器)     │
            │              │      └──────────────┘
            ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Scheduler Layer                      │   │
│  │                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │ Coordinator  │  │ MemoryManager│             │   │
│  │  │   (核心)     │◄─│ (PagedAttn)  │             │   │
│  │  └──────┬───────┘  └──────┬───────┘             │   │
│  │         │                  │                        │   │
│  │         │                  ▼                        │   │
│  │         │    ┌─────────────────────┐              │   │
│  │         │    │    RadixTree      │              │   │
│  │         │    │   (前缀缓存)       │              │   │
│  │         │    └─────────────────────┘              │   │
│  │         │                                        │   │
│  │  ┌──────▼───────┐                               │   │
│  │  │  Scheduling  │                               │   │
│  │  │   Policy     │                               │   │
│  │  └──────────────┘                               │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
            │
            │ WorkerCommand
            ▼
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Worker Layer                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │   │
│  │  │   Model    │  │  KV Cache  │  │   CUDA     │ │   │
│  │  │  (LLaMA3)  │  │  (GPU)     │  │  Kernel   │ │   │
│  │  └────────────┘  └────────────┘  └────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 核心组件关系

```text
                    ┌──────────────────┐
                    │  Server / Client │
                    └────────┬─────────┘
                             │ SchedulerCommand::AddRequest
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  Coordinator                          │
│                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │  Ingest  │   │ Schedule │   │  Apply   │       │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘       │
│       │              │              │               │
│       │              ▼              │               │
│       │      ┌──────────────┐      │               │
│       │      │   Policy     │      │               │
│       │      │ Continuous   │      │               │
│       │      └──────────────┘      │               │
│       │              │              │               │
│       │              │              ▼               │
│       │              │    ┌──────────────────┐   │
│       │              │    │ MemoryManager   │   │
│       │              │    ├────────────────┤   │
│       │              │    │ BlockAllocator │   │
│       │              │    │ RadixTree     │   │
│       │              │    └──────────────────┘   │
│       │              │              │               │
│       │              │              ▼               │
│       │              │      ┌──────────┐          │
│       │              │      │ Execute  │          │
│       │              │      └────┬─────┘          │
│       │              │           │                │
│       └──────────────┴───────────┤                │
│                                    ▼                │
│                         ┌──────────────────┐       │
│                         │  WorkerProxy     │       │
│                         └────────┬─────────┘       │
└──────────────────────────────────┼───────────────────┘
                                   │ WorkerCommand::Forward
                                   ▼
                          ┌─────────────────┐
                          │     Worker      │
                          │  (GPU Model)    │
                          └─────────────────┘
```

---

## 核心模块

### 1. Coordinator (`coordinator.rs`)

**职责**: 系统核心协调器，实现五阶段调度循环。

**核心结构**:
```rust
pub struct Coordinator {
    policy: Box<dyn SchedulingPolicy>,    // 调度策略
    worker: WorkerProxy,                  // Worker 通信代理
    frontend: FrontendReceiver,            // 前端请求接收器
    memory: MemoryManager,                // 内存管理器
    state: GlobalState,                  // 全局状态
    config: CoordinatorConfig,            // 配置
}
```

**关键方法**:
```rust
impl Coordinator {
    pub fn new(
        policy: Box<dyn SchedulingPolicy>,
        worker: WorkerProxy,
        frontend: FrontendReceiver,
        config: CoordinatorConfig,
    ) -> Self;

    pub async fn run(&mut self) {
        loop {
            self.ingest_requests().await;
            let plan = self.schedule_requests();
            self.apply_plan(&plan).await?;
            self.execute_batch(&plan).await?;
            self.post_process(&plan).await;
        }
    }

    // 设置默认 Worker ID（用于单 Worker 模式）
    pub fn set_default_worker(&mut self, worker_id: String);
}
```

**生命周期**:
1. 创建 → 初始化所有组件
2. 运行 → 进入主循环
3. Ingest → 接收新请求
4. Schedule → 调用策略生成计划
5. Apply → 应用计划（抢占、分配）
6. Execute → 调用 Worker
7. Post-Process → 处理结果
8. 重复 3-7

---

### 2. MemoryManager (`memory/mod.rs`)

**职责**: 统一的内存管理器，整合 BlockAllocator、BlockTableManager 和 RadixTree。

**核心结构**:
```rust
pub struct MemoryManager {
    allocator: BlockAllocator,      // 物理块分配器
    block_tables: BlockTableManager, // 序列块表管理器
    radix_tree: Option<RadixTree>, // 前缀缓存树
    config: MemoryConfig,          // 配置
}
```

**关键方法**:
```rust
impl MemoryManager {
    pub fn new(config: MemoryConfig) -> Self;

    /// 为序列分配内存块
    pub fn allocate_for_sequence(
        &mut self,
        seq_id: SequenceId,
        num_blocks: usize,
    ) -> Result<(), String>;

    /// 释放序列内存
    pub fn free_sequence(
        &mut self,
        seq_id: &SequenceId,
    ) -> Result<(), String>;

    /// 拷贝序列（用于 Beam Search）
    pub fn fork_sequence(
        &mut self,
        src_seq_id: &SequenceId,
        new_seq_id: SequenceId,
    ) -> Result<(), String>;

    /// 匹配前缀（Prompt Caching）
    pub fn match_prefix(
        &mut self,
        tokens: &[i32],
    ) -> Option<PrefixMatch>;
}
```

**内存分配流程**:
```text
1. allocate_for_sequence(seq_id, num_blocks)
   ↓
2. 检查 RadixTree 是否有缓存前缀
   ↓
3. 如果有缓存:
   - 复用缓存的物理块
   - incref_prefix() 增加引用计数
   - 只需分配剩余部分
   ↓
4. 如果没有缓存:
   - 从 BlockAllocator 分配新块
   - 更新 BlockTable
   - 插入 RadixTree
```

---

### 3. BlockAllocator (`memory/allocator.rs`)

**职责**: PagedAttention 的物理块管理器。

**核心结构**:
```rust
pub struct BlockAllocator {
    free_list: Vec<PhysicalBlockId>,  // 空闲块链表
    block_refs: HashMap<PhysicalBlockId, usize>,  // 引用计数
    total_blocks: usize,
    used_blocks: usize,
}
```

**关键方法**:
```rust
impl BlockAllocator {
    /// 分配物理块（返回连续的块列表）
    pub fn allocate(&mut self, count: usize) -> Vec<PhysicalBlockId>;

    /// 释放物理块
    pub fn free(&mut self, block_ids: &[PhysicalBlockId]);

    /// 增加引用计数（CoW 优化）
    pub fn incref(&mut self, block_id: PhysicalBlockId);

    /// 减少引用计数，返回是否释放
    pub fn decref(&mut self, block_id: PhysicalBlockId) -> bool;

    /// 获取统计信息
    pub fn stats(&self) -> AllocatorStats;
}
```

**分配策略**:
- 优先使用 free_list 中的块（O(1) 查找）
- 支持 CoW：多个序列可以共享物理块
- 引用计数为 0 时自动回收

---

### 4. RadixTree (`memory/radix_tree.rs`)

**职责**: 前缀缓存，支持 Prompt Caching。

**核心结构**:
```rust
pub struct RadixNode {
    children: HashMap<i32, Box<RadixNode>>,  // 子节点
    tokens: Vec<i32>,                        // 当前节点的 tokens
    blocks: Vec<PhysicalBlockId>,            // 对应的物理块
    ref_count: usize,                        // 引用计数（用于 LRU）
}

pub struct RadixTree {
    root: RadixNode,
    max_nodes: usize,
    current_nodes: usize,
}
```

**关键方法**:
```rust
impl RadixTree {
    /// 查找最长匹配的前缀
    pub fn match_prefix(&mut self, tokens: &[i32]) -> Option<PrefixMatch> {
        // PrefixMatch {
        //     matched_tokens: Vec<i32>,
        //     matched_blocks: Vec<PhysicalBlockId>,
        //     handle: RadixHandle,
        // }
    }

    /// 插入新的前缀
    pub fn insert(
        &mut self,
        tokens: &[i32],
        blocks: &[PhysicalBlockId],
    ) -> Result<(), String>;

    /// 增加引用计数
    pub fn incref(&mut self, handle: RadixHandle);

    /// 减少引用计数
    pub fn decref(&mut self, handle: RadixHandle);

    /// LRU 驱逐（当节点数超过 max_nodes 时）
    pub fn evict_lru(&mut self, count: usize) -> usize;
}
```

**前缀匹配示例**:
```text
请求 tokens: [A, B, C, D, E, F]
RadixTree:
  [root]
    └── [A, B, C] → blocks [0, 1, 2]  (ref_count: 2)
          └── [D, E] → blocks [3, 4]     (ref_count: 1)

匹配结果:
  - 匹配: [A, B, C, D, E] → blocks [0, 1, 2, 3, 4]
  - 新增: [F] → 需要分配新 block [5]
  - 节省: 5/6 = 83% 的计算量
```

---

### 5. ContinuousBatchingPolicy (`policy/continuous.rs`)

**职责**: 连续批处理策略，优先保证 Decode 任务。

**核心逻辑**:
```rust
impl SchedulingPolicy for ContinuousBatchingPolicy {
    fn schedule(&self, context: &ScheduleContext) -> BatchPlan {
        let mut plan = BatchPlan::default();

        // Step 1: 优先处理 Decode 任务（已有 KV Cache）
        for seq in context.running_sequences() {
            if plan.can_add_decode() {
                plan.add_decode(seq.id());
            }
        }

        // Step 2: 处理 Prefill 任务（新请求或 Swapped）
        for seq in context.waiting_sequences() {
            if plan.can_add_prefill() {
                // 检查 RadixTree 前缀匹配
                if let Some(prefix) = context.match_prefix(&seq.tokens) {
                    plan.add_prefill_with_cache(seq.id(), prefix);
                } else {
                    plan.add_prefill(seq.id());
                }
            }
        }

        // Step 3: 必要时抢占低优先级请求
        if plan.need_memory() {
            plan.preempt_low_priority();
        }

        plan
    }
}
```

**性能特性**:
- Decode 任务优先，提高 Throughput
- 前缀缓存复用，减少 Prefill 时间
- 支持抢占，保证高优先级请求响应

---

### 6. WorkerProxy (`transport/worker_proxy.rs`)

**职责**: 与 Worker 通信的代理，管理连接状态。

**核心结构**:
```rust
pub struct WorkerProxy {
    endpoint: String,                     // ZMQ endpoint
    socket: zmq::Socket,                  // ZMQ Router socket
    workers: HashMap<String, WorkerInfo>,  // 注册的 Workers
    connections: HashMap<String, ConnectionState>,
    timeout_ms: u64,
}
```

**关键方法**:
```rust
impl WorkerProxy {
    pub async fn new(
        endpoint: String,
        timeout_ms: u64,
    ) -> Result<Self>;

    /// 等待 Worker 注册
    pub async fn wait_for_registration(&mut self) -> Result<WorkerInfo>;

    /// 加载模型
    pub async fn load_model(
        &mut self,
        worker_id: &str,
        params: ModelLoadParams,
    ) -> Result<ModelInfo>;

    /// 初始化 KV Cache
    pub async fn init_kv_cache(
        &mut self,
        worker_id: &str,
        params: InitKVCacheParams,
    ) -> Result<KVCacheInfo>;

    /// 执行推理
    pub async fn forward(
        &mut self,
        worker_id: &str,
        params: ForwardParams,
    ) -> Result<ForwardResult>;

    /// 健康检查
    pub async fn health_check(&mut self, worker_id: &str) -> Result<bool>;
}
```

**Worker 生命周期**:
```text
1. Worker 启动
   ↓
2. Worker 发送 Register 消息 → WorkerProxy
   ↓
3. WorkerProxy 返回 Register 确认
   ↓
4. Coordinator 调用 load_model() → Worker
   ↓
5. Coordinator 调用 init_kv_cache() → Worker
   ↓
6. Worker 就绪，开始接收 forward 请求
   ↓
7. 定期 health_check()
```

---

### 7. GlobalState (`state.rs`)

**职责**: 全局状态管理，维护请求队列和序列状态。

**核心结构**:
```rust
pub struct GlobalState {
    // 三种队列
    waiting_queue: VecDeque<SequenceId>,     // 等待队列
    running_queue: Vec<SequenceId>,         // 运行队列
    swapped_queue: Vec<SequenceId>,        // 被抢占队列

    // 序列映射
    sequences: HashMap<SequenceId, Sequence>,

    // 请求 ID 映射
    request_map: HashMap<String, SequenceId>,

    // 输出通道
    output_channels: HashMap<SequenceId, OutputChannels>,
}
```

**Sequence 状态**:
```rust
pub struct Sequence {
    pub request_id: String,
    pub seq_id: SequenceId,
    pub input_tokens: Vec<i32>,
    pub generated_tokens: Vec<i32>,
    pub allocated_blocks: Vec<PhysicalBlockId>,
    pub state: RequestState,  // Waiting | Running | Swapped | Finished
    pub priority: i32,

    // 输出通道
    pub output_tx: Option<mpsc::UnboundedSender<StepOutput>>,
    pub finish_tx: Option<mpsc::UnboundedSender<FinishOutput>>,
}
```

**状态转换**:
```text
Waiting (new request)
   ↓ add_request()
Running (scheduled)
   ↓ preempt()
Swapped (evicted)
   ↓ resume()
Running
   ↓ finish()
Finished
   ↓ free_sequence()
Removed
```

---

## 通信协议

### Server ↔ Scheduler 协议

**传输方式**: ZeroMQ PUSH/PULL
**序列化**: MessagePack (rmp-serde)
**地址**: 可配置，默认 `ipc:///tmp/rustinfer.ipc`（需要修改）

**Server → Scheduler (请求)**:
```rust
pub enum SchedulerCommand {
    AddRequest(InferRequest),
    AbortRequest(String),  // request_id
}

pub struct InferRequest {
    pub request_id: String,
    pub input_tokens: Vec<i32>,  // Token IDs
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub stream: bool,
}
```

**Scheduler → Server (响应 - 流式)**:
```rust
pub enum SchedulerOutput {
    Step(StepOutput),      // 单个 token 生成
    Finish(FinishOutput),  // 完成通知
    Error(ErrorOutput),    // 错误
}

pub struct StepOutput {
    pub request_id: String,
    pub token_id: u32,
    pub position: usize,
    pub is_first: bool,
}

pub struct FinishOutput {
    pub request_id: String,
    pub generated_tokens: Vec<u32>,
    pub finish_reason: String,  // "eos" | "length" | "abort"
    pub usage: UsageStats,
}
```

**消息格式**:
```rust
// 发送（Server → Scheduler）
let data = rmp_serde::to_vec(&SchedulerCommand::AddRequest(request))?;
zmq_socket.send(&data, 0)?;

// 接收（Scheduler → Server）
let data = zmq_socket.recv_bytes(0)?;
let output: SchedulerOutput = rmp_serde::from_slice(&data)?;
```

---

### Scheduler ↔ Worker 协议

**传输方式**: ZeroMQ ROUTER/DEALER (Actor 模式)
**序列化**: MessagePack
**地址**: `ipc:///tmp/rustinfer-scheduler.ipc`

**Scheduler → Worker (命令)**:
```rust
pub enum WorkerCommand {
    Register(WorkerInfo),
    LoadModel(ModelLoadParams),
    InitKVCache(InitKVCacheParams),
    Profile(ProfileParams),
    Forward(ForwardParams),
    HealthCheck,
}

pub struct ForwardParams {
    pub seq_id: SequenceId,
    pub input_tokens: Vec<i32>,
    pub prefill_indices: Vec<usize>,    // KV Cache 中已缓存的索引
    pub new_indices: Vec<usize>,        // 需要计算的索引
    pub sampling_params: SamplingParams,
}
```

**Worker → Scheduler (响应)**:
```rust
pub enum WorkerResponse {
    RegisterAck,
    LoadModelCompleted(ModelInfo),
    InitKVCacheCompleted(KVCacheInfo),
    ProfileCompleted(ProfileResult),
    ForwardCompleted(ForwardResult),
    HealthCheckResponse(bool),
    Error(ErrorMessage),
}

pub struct ForwardResult {
    pub seq_id: SequenceId,
    pub next_token: u32,
    pub logits: Vec<f32>,  // 可选，用于调试
    pub computed_tokens: usize,
}
```

---

## 内存管理

### PagedAttention 设计

**核心思想**: 将 KV Cache 划分为固定大小的物理块，实现灵活的内存管理。

**数据结构**:
```rust
// 物理块（GPU 上的连续内存）
PhysicalBlock {
    size: block_size (16 tokens),
    K: [num_heads, block_size, head_dim],
    V: [num_heads, block_size, head_dim],
}

// 逻辑块（序列视角的逻辑视图）
LogicalBlock {
    physical_id: PhysicalBlockId,
    ref_count: usize,  // 支持共享
}

// BlockTable（映射关系）
BlockTable {
    logical_blocks: [PhysicalBlockId; sequence_length],
}
```

**内存分配示例**:
```text
序列 A: tokens [0, 1, 2, ..., 47]
  逻辑块: [L0, L1, L2]
  物理块: [P10, P11, P12]

序列 B: tokens [0, 1, 2, ..., 31] (前缀相同)
  逻辑块: [L0, L1]
  物理块: [P10, P11]  ← 复用序列 A 的块

物理块状态:
  P10: ref_count = 2  (A 和 B 共享)
  P11: ref_count = 2
  P12: ref_count = 1
  P13: free
  ...
```

---

### RadixTree 前缀缓存

**树结构**:
```text
[root]
  ├── [A, B, C] → blocks [0, 1, 2]
  │     ├── [D, E] → blocks [3, 4]
  │     │     ├── [F, G] → blocks [5, 6] (ref: 1)
  │     │     └── [F, H] → blocks [5, 7] (ref: 1)
  │     │              ↑
  │     │         共享 block [5]
  │     └── [I, J] → blocks [8, 9] (ref: 1)
  │
  └── [X, Y, Z] → blocks [10, 11] (ref: 2)
```

**匹配流程**:
```rust
// 请求: [A, B, C, D, E, K]
let result = radix_tree.match_prefix(tokens);

// 返回:
PrefixMatch {
    matched_tokens: [A, B, C, D, E],  // 匹配的前缀
    matched_blocks: [0, 1, 2, 3, 4],  // 对应的物理块
    handle: RadixHandle,                // 用于引用计数
}

// 只需要为 [K] 分配新块
let new_blocks = allocator.allocate(1);
radix_tree.insert([A, B, C, D, E, K], [0,1,2,3,4, new_block]);
```

---

### 内存驱逐策略

**LRU (Least Recently Used)**:
```rust
pub struct EvictionPolicy {
    lru_list: VecDeque<RadixNodeRef>,
    node_timestamps: HashMap<usize, Instant>,
}

impl EvictionPolicy {
    pub fn on_access(&mut self, node: &mut RadixNode) {
        // 更新访问时间
        node.last_access_time = Instant::now();
    }

    pub fn select_eviction_candidates(
        &mut self,
        tokens_to_evict: usize,
    ) -> Vec<*mut RadixNode> {
        // 选择最久未使用的节点
        self.lru_list
            .iter()
            .take(tokens_to_evict)
            .map(|n| n.as_ptr())
            .collect()
    }
}
```

**驱逐条件**:
1. RadixTree 节点数超过 `max_nodes`
2. 物理块使用率超过 `gpu_memory_utilization`

**驱逐策略**:
1. 优先驱逐引用计数为 0 的节点
2. 使用 LRU 选择最久未使用的节点
3. 驱逐前检查是否有活跃请求在使用

---

## 调度策略

### ScheduleContext

**调度上下文**: 策略决策的输入。

```rust
pub struct ScheduleContext<'a> {
    pub waiting: Vec<&'a Sequence>,       // 等待队列
    pub running: Vec<&'a Sequence>,       // 运行队列
    pub swapped: Vec<&'a Sequence>,       // 被抢占队列

    pub memory_stats: MemoryStats,        // 内存统计
    pub block_size: usize,
    pub max_batch_size: usize,
    pub max_tokens_per_step: usize,
}

impl<'a> ScheduleContext<'a> {
    /// 匹配前缀（Prompt Caching）
    pub fn match_prefix(&self, tokens: &[i32]) -> Option<PrefixMatch> {
        self.radix_tree.as_ref()?.match_prefix(tokens)
    }

    /// 检查是否可以添加 Prefill
    pub fn can_add_prefill(&self, tokens_len: usize) -> bool {
        let required_blocks = (tokens_len + self.block_size - 1) / self.block_size;
        required_blocks <= self.memory_stats.free_blocks
    }
}
```

---

### BatchPlan

**批次计划**: 策略的输出。

```rust
pub struct BatchPlan {
    pub prefill: Vec<PrefillTask>,      // Prefill 任务列表
    pub decode: Vec<DecodeTask>,        // Decode 任务列表
    pub preempt: Vec<PreemptTask>,      // 抢占任务列表
    pub estimated_memory: usize,         // 预估内存使用
    pub total_tokens: usize,             // 总 token 数
}

pub struct PrefillTask {
    pub seq_id: SequenceId,
    pub tokens: Vec<i32>,
    pub cached_blocks: Option<Vec<PhysicalBlockId>>,  // 前缀缓存
    pub new_blocks_needed: usize,
}

pub struct DecodeTask {
    pub seq_id: SequenceId,
    pub num_tokens: usize,
}

pub struct PreemptTask {
    pub seq_id: SequenceId,
    pub reason: PreemptReason,  // "memory" | "priority"
}
```

---

### ContinuousBatchingPolicy

**调度逻辑**:
```rust
fn schedule(&self, ctx: &ScheduleContext) -> BatchPlan {
    let mut plan = BatchPlan::default();

    // ========== Phase 1: Decode 任务 (优先) ==========
    for seq in &ctx.running {
        if !self.can_add_decode(&plan, ctx.max_batch_size) {
            break;
        }
        plan.add_decode(DecodeTask {
            seq_id: seq.id(),
            num_tokens: 1,  // Decode 每次生成 1 个 token
        });
    }

    // ========== Phase 2: Prefill 任务 ==========
    for seq in &ctx.waiting {
        if !self.can_add_prefill(&plan, ctx.max_batch_size) {
            break;
        }

        // 检查前缀缓存
        if let Some(prefix) = ctx.match_prefix(&seq.input_tokens) {
            plan.add_prefill(PrefillTask {
                seq_id: seq.id(),
                tokens: seq.input_tokens.clone(),
                cached_blocks: Some(prefix.blocks),
                new_blocks_needed: seq.input_tokens.len() - prefix.matched_len,
            });
        } else {
            plan.add_prefill(PrefillTask {
                seq_id: seq.id(),
                tokens: seq.input_tokens.clone(),
                cached_blocks: None,
                new_blocks_needed: (seq.input_tokens.len() + ctx.block_size - 1) / ctx.block_size,
            });
        }
    }

    // ========== Phase 3: 抢占 (如果内存不足) ==========
    if plan.estimated_memory > ctx.memory_stats.total_blocks {
        let to_preempt = self.select_preemption_candidates(
            &plan,
            &ctx.swapped,
            ctx.preemption_threshold,
        );
        for seq in to_preempt {
            plan.add_preempt(PreemptTask {
                seq_id,
                reason: PreemptReason::Memory,
            });
        }
    }

    plan
}
```

**性能特性**:
- Decode 优先：保证已运行的请求继续完成
- 前缀缓存：减少 Prefill 计算量
- 抢占机制：高优先级请求优先

---

## 状态管理

### Sequence 状态机

```text
┌───────────┐
│  Waiting  │ ← 新请求到达
└─────┬─────┘
      │ schedule()
      ▼
┌───────────┐
│  Running  │ ← 已分配内存，正在执行
└─────┬─────┘
      │ preempt()
      ▼
┌───────────┐
│  Swapped  │ ← 被抢占，等待恢复
└─────┬─────┘
      │ resume()
      ▼
┌───────────┐
│  Running  │
└─────┬─────┘
      │ finish()
      ▼
┌───────────┐
│ Finished  │ ← 完成
└─────┬─────┘
      │ free_sequence()
      ▼
┌───────────┐
│  Removed  │ ← 从状态中移除
└───────────┘
```

**状态转换方法**:
```rust
impl GlobalState {
    pub fn add_request(&mut self, req: InferRequest) -> SequenceId {
        let seq_id = self.next_seq_id();
        let seq = Sequence::new(req, seq_id);
        self.sequences.insert(seq_id, seq);
        self.waiting_queue.push_back(seq_id);
        seq_id
    }

    pub fn transition_to_running(&mut self, seq_id: SequenceId) {
        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.state = RequestState::Running;
        self.waiting_queue.retain(|&id| id != seq_id);
        self.running_queue.push(seq_id);
    }

    pub fn preempt(&mut self, seq_id: SequenceId) {
        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.state = RequestState::Swapped;
        self.running_queue.retain(|&id| id != seq_id);
        self.swapped_queue.push_back(seq_id);
    }

    pub fn finish(&mut self, seq_id: SequenceId) {
        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.state = RequestState::Finished;
        self.running_queue.retain(|&id| id != seq_id);
    }

    pub fn remove(&mut self, seq_id: &SequenceId) {
        self.sequences.remove(seq_id);
        self.waiting_queue.retain(|id| id != seq_id);
        self.running_queue.retain(|id| id != seq_id);
        self.swapped_queue.retain(|id| id != seq_id);
    }
}
```

---

### 输出通道管理

**流式输出**: 每个序列有独立的输出通道。

```rust
pub struct OutputChannels {
    pub step_tx: mpsc::UnboundedSender<StepOutput>,
    pub finish_tx: mpsc::UnboundedSender<FinishOutput>,
}

impl GlobalState {
    pub fn create_output_channels(
        &mut self,
        seq_id: SequenceId,
    ) -> (mpsc::UnboundedReceiver<StepOutput>, mpsc::UnboundedReceiver<FinishOutput>) {
        let (step_tx, step_rx) = mpsc::unbounded_channel();
        let (finish_tx, finish_rx) = mpsc::unbounded_channel();

        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.output_tx = Some(step_tx);
        seq.finish_tx = Some(finish_tx);

        (step_rx, finish_rx)
    }

    pub fn send_step_output(&self, seq_id: SequenceId, token_id: u32) {
        if let Some(seq) = self.sequences.get(&seq_id) {
            if let Some(ref tx) = seq.output_tx {
                let _ = tx.send(StepOutput {
                    request_id: seq.request_id.clone(),
                    token_id,
                    position: seq.generated_tokens.len(),
                    is_first: seq.generated_tokens.is_empty(),
                });
            }
        }
    }
}
```

---

## 配置系统

### SchedulerConfig

**顶层配置结构**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Parser)]
pub struct SchedulerConfig {
    pub network: NetworkConfig,
    pub model: ModelConfig,
    pub memory: MemoryConfig,
    pub scheduling: SchedulingConfig,
    pub parallelism: ParallelismConfig,
    pub logging: LoggingConfig,
    pub config_file: Option<PathBuf>,
}
```

**加载方式**:
```rust
// 从命令行参数加载
let config = SchedulerConfig::from_args();

// 从配置文件加载
let config = SchedulerConfig::from_file("scheduler.yaml")?;

// 混合加载（命令行覆盖配置文件）
let config = SchedulerConfig::load()?;
```

---

### NetworkConfig

**网络配置**:
```rust
pub struct NetworkConfig {
    /// ZeroMQ endpoint for Worker communication
    pub worker_endpoint: String,  // "ipc:///tmp/rustinfer-scheduler.ipc"

    /// Worker RPC timeout (milliseconds)
    pub worker_timeout_ms: u64,  // 30000

    /// Number of workers to wait for
    pub num_workers: usize,  // 1
}
```

---

### ModelConfig

**模型配置**:
```rust
pub struct ModelConfig {
    /// Model path (safetensors format)
    pub model_path: String,  // "/path/to/model"

    /// Model dtype (bf16, fp16, fp32)
    pub dtype: String,  // "bf16"

    /// Enable Flash Attention
    pub enable_flash_attn: bool,  // true

    /// Optional custom config overrides (JSON string)
    pub custom_config: Option<String>,
}
```

**注意**: Scheduler 不持有 Tokenizer，只使用 `model_path` 读取 `config.json` 中的元数据。

---

### MemoryConfig

**内存配置**:
```rust
pub struct MemoryConfig {
    /// Block size (number of tokens per block)
    pub block_size: usize,  // 16 (vLLM 默认)

    /// Total number of GPU blocks
    pub total_blocks: usize,  // 0 = auto (profile 后自动计算)

    /// GPU memory utilization ratio (0.0 - 1.0)
    pub gpu_memory_utilization: f32,  // 0.9

    /// Enable prefix caching (RadixTree)
    pub enable_prefix_cache: bool,  // true

    /// Prefix cache capacity (max tokens)
    pub prefix_cache_capacity: usize,  // 100000

    /// Enable Copy-on-Write (for Beam Search)
    pub enable_cow: bool,  // false
}
```

**自动计算 total_blocks**:
```rust
if config.total_blocks == 0 {
    // Profile GPU 显存
    let profile = worker.profile(&worker_id, params).await?;

    // 计算每个 block 需要的显存
    let bytes_per_block = block_size
        * num_layers
        * num_kv_heads
        * head_dim
        * 2  // K 和 V
        * dtype_bytes;

    // 计算可用显存可容纳的 block 数
    let computed_blocks = profile.available_kv_cache_memory / bytes_per_block;

    // 应用 gpu_memory_utilization 系数
    config.total_blocks = (computed_blocks as f32 * config.gpu_memory_utilization) as usize;
}
```

---

### SchedulingConfig

**调度配置**:
```rust
pub struct SchedulingConfig {
    /// Scheduling policy name
    pub policy: String,  // "continuous" (default)

    /// Max batch size (number of concurrent requests)
    pub max_batch_size: usize,  // 256

    /// Max tokens per step (防止单次计算过大)
    pub max_tokens_per_step: usize,  // 4096

    /// Enable preemption (抢占低优先级请求)
    pub enable_preemption: bool,  // true

    /// Preemption threshold (free blocks)
    pub preemption_threshold: usize,  // 0

    /// Enable swap to CPU memory
    pub enable_swap: bool,  // false

    /// Default request priority
    pub default_priority: i32,  // 0

    /// Idle sleep duration (milliseconds)
    pub idle_sleep_ms: u64,  // 1
}
```

---

### ParallelismConfig

**并行配置**:
```rust
pub struct ParallelismConfig {
    /// Tensor Parallel size
    pub tp_size: usize,  // 1

    /// Pipeline Parallel size
    pub pp_size: usize,  // 1

    /// Tensor Parallel rank (auto-assigned by Worker)
    pub tp_rank: Option<usize>,

    /// Pipeline Parallel rank (auto-assigned by Worker)
    pub pp_rank: Option<usize>,
}
```

---

## API 参考

### 启动 Scheduler

**二进制**: `cargo run -p infer-scheduler --bin infer-scheduler`

**命令行参数**:
```bash
# 最简启动
rustinfer-scheduler \
    --model-path /path/to/model

# 完整配置
rustinfer-scheduler \
    --config-file scheduler.yaml \
    --model-path /mnt/models/llama3-8b \
    --dtype bf16 \
    --block-size 16 \
    --max-batch-size 32 \
    --worker-endpoint ipc:///tmp/rustinfer-scheduler.ipc \
    --num-workers 1 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-cache

# 使用配置文件
rustinfer-scheduler --config-file scheduler.yaml
```

**配置文件示例** (`scheduler.yaml`):
```yaml
# Network
worker_endpoint: "ipc:///tmp/rustinfer-scheduler.ipc"
worker_timeout_ms: 30000
num_workers: 1

# Model
model_path: "/mnt/models/llama3-8b"
dtype: "bf16"
enable_flash_attn: true

# Memory
block_size: 16
total_blocks: 0  # auto
gpu_memory_utilization: 0.9
enable_prefix_cache: true
prefix_cache_capacity: 100000
enable_cow: false

# Scheduling
policy: "continuous"
max_batch_size: 256
max_tokens_per_step: 4096
enable_preemption: true
preemption_threshold: 0
enable_swap: false
default_priority: 0
idle_sleep_ms: 1

# Parallelism
tp_size: 1
pp_size: 1

# Logging
log_level: "info"
log_format: "text"
log_to_file: false
log_file: "/tmp/rustinfer-scheduler.log"
```

---

### 编程接口

**使用 Scheduler 作为库**:

```rust
use infer_scheduler::{SchedulerConfig, Coordinator};
use infer_scheduler::policy::ContinuousBatchingPolicy;
use infer_scheduler::transport::{create_frontend_channel, WorkerProxy};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 加载配置
    let config = SchedulerConfig::load()?;

    // 2. 创建前端通道
    let (frontend_tx, frontend_rx) = create_frontend_channel();

    // 3. 创建 Worker 代理
    let worker_proxy = WorkerProxy::new(
        config.network.worker_endpoint.clone(),
        config.network.worker_timeout_ms,
    ).await?;

    // 4. 等待 Worker 注册
    let worker_info = worker_proxy.wait_for_registration().await?;

    // 5. 初始化 Worker
    worker_proxy.load_model(&worker_info.worker_id, model_params).await?;
    worker_proxy.init_kv_cache(&worker_info.worker_id, kv_params).await?;

    // 6. 创建调度策略
    let policy = Box::new(ContinuousBatchingPolicy::new(
        config.to_policy_config()
    ));

    // 7. 创建 Coordinator
    let mut coordinator = Coordinator::new(
        policy,
        worker_proxy,
        frontend_rx,
        config.to_coordinator_config(),
    );

    coordinator.set_default_worker(worker_info.worker_id);

    // 8. 启动调度循环
    coordinator.run().await;

    Ok(())
}
```

---

### 发送推理请求

**通过 Channel 发送**:
```rust
use infer_protocol::{SchedulerCommand, InferRequest, SamplingParams};

// 创建请求
let request = InferRequest {
    request_id: "req-001".to_string(),
    input_tokens: vec![1, 2, 3, 4, 5],  // Token IDs
    max_tokens: 100,
    sampling_params: SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
    },
    stream: true,
    priority: 0,
};

// 发送给 Scheduler
frontend_tx.send(SchedulerCommand::AddRequest(request))?;
```

---

### 扩展指南

### 添加新的调度策略

**步骤**:

1. **实现策略接口**:
```rust
// policy/custom.rs
use crate::policy::{SchedulingPolicy, ScheduleContext, BatchPlan};

pub struct CustomPolicy {
    config: SchedulePolicyConfig,
}

impl SchedulingPolicy for CustomPolicy {
    fn schedule(&self, ctx: &ScheduleContext) -> BatchPlan {
        // 你的调度逻辑
        let mut plan = BatchPlan::default();

        // 示例: 按优先级排序
        let mut waiting: Vec<_> = ctx.waiting.iter().collect();
        waiting.sort_by(|a, b| b.priority.cmp(&a.priority));

        for seq in waiting {
            if plan.can_add_prefill() {
                plan.add_prefill(PrefillTask {
                    seq_id: seq.id(),
                    tokens: seq.input_tokens.clone(),
                    cached_blocks: ctx.match_prefix(&seq.input_tokens).map(|p| p.blocks),
                    new_blocks_needed: /* ... */,
                });
            }
        }

        plan
    }

    fn name(&self) -> &'static str {
        "custom"
    }

    fn config(&self) -> &SchedulePolicyConfig {
        &self.config
    }
}
```

2. **注册策略**:
```rust
// policy/mod.rs
pub mod continuous;
pub mod custom;

use continuous::ContinuousBatchingPolicy;
use custom::CustomPolicy;

pub fn create_policy(name: &str, config: SchedulePolicyConfig) -> Box<dyn SchedulingPolicy> {
    match name {
        "continuous" => Box::new(ContinuousBatchingPolicy::new(config)),
        "custom" => Box::new(CustomPolicy::new(config)),
        _ => panic!("Unknown policy: {}", name),
    }
}
```

3. **使用策略**:
```rust
let policy = create_policy(&config.scheduling.policy, config.to_policy_config());
```

---

### 添加新的驱逐策略

**步骤**:

1. **实现驱逐接口**:
```rust
// memory/eviction_custom.rs
use crate::memory::eviction::EvictionPolicy;
use crate::memory::radix_tree::RadixNode;

pub struct CustomEvictionPolicy {
    // 你的策略状态
}

impl EvictionPolicy for CustomEvictionPolicy {
    fn select_eviction_candidates(
        &mut self,
        tokens_to_evict: usize,
        nodes: &mut Vec<*mut RadixNode>,
    ) -> Vec<*mut RadixNode> {
        // 示例: LFU (Least Frequently Used)
        nodes.sort_by(|a, b| {
            let a_ref = unsafe { &**a };
            let b_ref = unsafe { &**b };
            a_ref.hit_count.cmp(&b_ref.hit_count)
        });
        nodes.into_iter().take(tokens_to_evict).collect()
    }

    fn on_access(&mut self, node: &mut RadixNode) {
        node.hit_count += 1;
    }

    fn on_insert(&mut self, node: &mut RadixNode) {
        node.hit_count = 1;
    }
}
```

2. **使用策略**:
```rust
use crate::memory::eviction::CustomEvictionPolicy;

let eviction_policy = Box::new(CustomEvictionPolicy::new());
let radix_tree = RadixTree::with_eviction(eviction_policy);
```

---

### 添加新的通信协议

**示例: 使用 gRPC 替代 ZeroMQ**

1. **定义 protobuf**:
```protobuf
// scheduler.proto
service Scheduler {
    rpc AddRequest(InferRequest) returns (stream StepOutput);
    rpc AbortRequest(AbortRequest) returns (Empty);
}

message InferRequest {
    string request_id = 1;
    repeated int32 input_tokens = 2;
    // ...
}
```

2. **实现 gRPC 服务**:
```rust
// transport/grpc_server.rs
use tonic::transport::Server;

pub struct GrpcServer {
    frontend_tx: mpsc::UnboundedSender<SchedulerCommand>,
}

#[tonic::async_trait]
impl Scheduler for GrpcServer {
    type AddRequestStream = ReceiverStream<StepOutput>;

    async fn add_request(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<Self::AddRequestStream>, Status> {
        let req = request.into_inner();

        // 转换为 SchedulerCommand
        let cmd = SchedulerCommand::AddRequest(InferRequest {
            request_id: req.request_id,
            input_tokens: req.input_tokens,
            // ...
        });
        self.frontend_tx.send(cmd)?;

        // 创建流式响应通道
        let (tx, rx) = mpsc::channel(100);
        self.output_map.insert(req.request_id.clone(), tx);

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

---

## 最佳实践

### 1. 内存管理

**推荐配置**:
```yaml
memory:
  block_size: 16  # vLLM 默认值
  total_blocks: 0  # 让 scheduler 自动计算
  gpu_memory_utilization: 0.9  # 留 10% 显存给模型权重
  enable_prefix_cache: true  # 对话场景必开
```

**计算公式**:
```
KV Cache 显存 = total_blocks × block_size × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes

示例 (Llama3-8B):
- total_blocks: 1000
- block_size: 16
- num_layers: 32
- num_kv_heads: 8
- head_dim: 128
- dtype: bf16 (2 bytes)

= 1000 × 16 × 32 × 8 × 128 × 2 × 2
≈ 13.1 GB
```

---

### 2. 调度策略选择

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 高 Throughput | `continuous` | 优先 Decode，充分利用 GPU |
| 低延迟 | `priority` | 优先处理新请求 |
| 公平性 | `fair` | 按请求时间公平调度 |
| 多租户 | `custom` | 支持租户隔离 |

---

### 3. 前缀缓存优化

**适用场景**:
- ✅ 对话系统（System Prompt 复用）
- ✅ 代码生成（重复代码片段）
- ✅ 文档分析（重复章节）

**不适用场景**:
- ❌ 随机噪声输入（缓存命中率低）
- ❌ 高度个性化请求（前缀不重合）

**参数调优**:
```yaml
memory:
  prefix_cache_capacity: 100000  # 100K tokens
  # 如果显存充足，可以增大到 1M tokens
```

---

### 4. 批次大小选择

**原则**: 平衡延迟和吞吐量

```yaml
scheduling:
  max_batch_size: 32  # 适用于大多数场景
  # 延迟敏感: 16-32
  # 吞吐量优先: 64-256
```

**性能参考**:
- Batch Size = 1: 最低延迟，最低吞吐量
- Batch Size = 32: 延迟增加 20%，吞吐量增加 5x
- Batch Size = 256: 延迟增加 3x，吞吐量增加 10x

---

### 5. 错误处理

**推荐模式**:
```rust
// Coordinator::execute_batch()
async fn execute_batch(&mut self, plan: &BatchPlan) -> Result<(), SchedulerError> {
    let results = self.worker.forward_batch(&plan.tasks).await;

    for (task, result) in plan.tasks.iter().zip(results) {
        match result {
            Ok(forward_result) => {
                self.handle_success(task, forward_result).await;
            }
            Err(e) => {
                tracing::error!("Forward failed for seq {}: {:?}", task.seq_id, e);
                self.handle_error(task, e).await;
            }
        }
    }

    Ok(())
}
```

---

### 6. 监控指标

**关键指标**:
```rust
pub struct SchedulerMetrics {
    // 请求统计
    pub total_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub preempted_requests: usize,

    // 内存统计
    pub memory_usage_percent: f64,
    pub cache_hit_rate: f64,

    // 队列统计
    pub waiting_queue_size: usize,
    pub running_queue_size: usize,

    // 性能统计
    pub avg_prefill_time_ms: f64,
    pub avg_decode_time_ms: f64,
    pub tokens_per_second: f64,
}
```

---

## 故障排查

### 问题 1: 内存不足 (OOM)

**症状**: Worker 崩溃，日志显示 "CUDA out of memory"

**解决方案**:
```yaml
memory:
  gpu_memory_utilization: 0.7  # 降低到 70%
  block_size: 16  # 减小块大小
  total_blocks: 500  # 手动设置较小的块数
```

---

### 问题 2: 缓存命中率低

**症状**: `cache_hit_rate < 30%`

**检查**:
1. 前缀缓存是否启用: `enable_prefix_cache: true`
2. 请求是否有重复前缀
3. RadixTree 容量是否足够

**优化**:
```yaml
memory:
  prefix_cache_capacity: 200000  # 增大缓存容量
```

---

### 问题 3: Worker 连接超时

**症状**: "Failed to register Worker: timeout"

**检查**:
1. Worker 是否正在运行
2. `worker_endpoint` 地址是否正确
3. IPC 文件权限是否正确

**调试**:
```bash
# 检查 IPC 文件
ls -la /tmp/rustinfer-scheduler.ipc

# 检查 Scheduler 是否在监听
ss -l | grep rustinfer

# 手动测试 ZMQ 连接
zmq-connect ipc:///tmp/rustinfer-scheduler.ipc
```

---

## 总结

RustInfer Scheduler 是一个高性能、可扩展的 LLM 推理调度器，核心特性包括：

### 优势
1. **纯 Token 层架构**: 避免文本传输开销
2. **PagedAttention 内存管理**: 高显存利用率
3. **前缀缓存**: 显著提升重复前缀的性能
4. **五阶段调度循环**: 逻辑清晰，易于优化
5. **可插拔策略**: 支持多种调度策略

### 适用场景
- 高并发 LLM 推理服务
- 需要低延迟的实时应用
- 资源受限的环境（显存优化）
- 多模型、多场景的推理平台

### 技术栈
- **异步运行时**: Tokio
- **通信**: ZeroMQ (Router/Dealer, Push/Pull)
- **序列化**: MessagePack
- **内存管理**: PagedAttention + RadixTree
- **并发**: Arc, RwLock, Mutex

### 性能目标
- **TTFT** (Time To First Token): < 100ms
- **Token Throughput**: > 50 tokens/s
- **Cache Hit Rate**: > 70% (对话场景)
- **GPU Utilization**: > 90%

---

**文档版本**: v0.1.0
**最后更新**: 2026-01-18
**维护者**: RustInfer Team
