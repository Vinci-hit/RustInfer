# RustInfer 高并发 KV 缓存系统开发人员指南

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [系统架构](#系统架构)
4. [关键组件](#关键组件)
5. [使用模式](#使用模式)
6. [并发模型](#并发模型)
7. [性能特性](#性能特性)
8. [故障排除](#故障排除)

---

## 概述

RustInfer 为大语言模型 (LLM) 实现了一个高性能、高并发的推理引擎。该系统经过优化，能够同时处理多个用户请求，同时高效地重用已缓存的计算结果。

### 解决什么问题？

在提供 LLM 服务时，相同的提示词前缀（prompt prefixes）经常出现在不同的请求中：
- “将英文翻译成西班牙文：”（出现在许多请求中）
- “问：你感觉怎么样？答：”（对话模板）
- 来自同一文档的上下文（摘要任务）

**如果没有缓存：** 每个请求都需要为整个提示词重新计算 KV 缓存。
**使用 RadixAttention：** 请求可以共享公共前缀的缓存计算结果。

### 为什么需要高并发？

现代 GPU 可以处理多个推理任务，但同步这些任务具有挑战性。RustInfer 使用细粒度锁来实现：
- 多个请求可以同时检查缓存
- GPU 前向传播（forward passes）逐个进行（确保正确性所必需）
- 最小化锁竞争，最大化吞吐量

---

## 核心概念

### 什么是 KV 缓存 (KV Cache)？

在 Transformer 模型中，过去 token 的 Key (K) 和 Value (V) 张量存储在内存中，而不是重新计算。这大大加快了推理速度。

```
示例：生成 "Hello, World!"

Token 0: "Hello"
  → 计算 "Hello" 的 K, V → 存储在缓存中

Token 1: ","
  → 从缓存中重用 "Hello" 的 K, V
  → 计算 "," 的新 K, V → 添加到缓存中

Token 2: "World"
  → 从缓存中重用 "Hello" 和 "," 的 K, V
  → 计算 "World" 的新 K, V → 添加到缓存中
```

### 什么是前缀共享 (Prefix Sharing)？

当多个请求的提示词存在重叠时，它们可以共享公共前缀的缓存 KV 值。

```
请求 A: "Translate English to Spanish: Hello"
请求 B: "Translate English to Spanish: Goodbye"
                    ↑ 公共前缀 ↑

两个请求都可以使用前缀对应的相同缓存 KV。
节省了公共部分的计算量！
```

### 什么是 RadixAttention？

RadixAttention 是一种前缀共享技术，它将缓存的 KV 值存储在 **基数树 (radix tree)**（一种压缩树结构）中。该树实现了：

1. **高效的前缀匹配** - 快速找到公共前缀
2. **内存重用** - 多个请求指向相同的缓存值
3. **自动清理** - 当内存满时，驱逐最近最少使用 (LRU) 的条目
 - [RadixTree](https://www.cs.usfca.edu/~galles/visualization/RadixTree.html) --可视化网站
---

## 系统架构

### 系统级概览

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         客户端请求 (通过 ZMQ)                              │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │ (msgpack 编码的 InferenceRequest)
                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        高并发 ZMQ 服务器                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ 每个请求都作为异步任务派生，以实现真正的并行                       │  │
│  │ 非阻塞的请求/响应处理                                              │  │
└──────────────────────────┬─────────────────────────────────────────────┘  │
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│              ConcurrentInferenceEngine (大脑/协调器)                       │
│                                                                            │
│  角色：调度工作，管理缓存决策                                              │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SharedRadixCache (Arc<RwLock<>>)                                    │ │
│  │  ─────────────────────────────────────────────────────────────────   │ │
│  │  功能：具有 O(1) LRU 驱逐机制的前缀共享 KV 缓存                    │ │
│  │  锁：RwLock = 允许多个读取者（缓存查询）或一个写入者               │ │
│  │  存储：KVPool 中物理 token 存储的索引                               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SharedKVPool (Arc<RwLock<>>)                                        │ │
│  │  ─────────────────────────────────────────────────────────────────   │ │
│  │  功能：K 和 V 张量的物理存储                                         │ │
│  │  锁：RwLock 用于并发访问                                             │ │
│  │  存储：GPU/CPU 上的实际张量数据                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  信号量 (Semaphore)：限制 N 个并发请求处理程序（默认：64）                │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                    CacheInstruction (缓存指令)
                 (cached_indices, new_indices)
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    Llama3 模型 (工作器/执行器)                            │
│                                                                            │
│  角色：使用缓存指令执行前向传播                                            │
│  锁：Mutex = 一次处理一个请求（GPU 是串行的）                             │
│                                                                            │
│  接收 CacheInstruction，指示：                                            │
│    • 哪些 KV 索引已缓存（读取）                                           │
│    • 哪些索引需要新的 KV 计算（写入）                                     │
│    • 结果写入位置（token 位置）                                            │
│                                                                            │
│  这是加载了权重的实际 LLM 模型                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 请求流程图

```
客户端提交 "Translate: Hello"
     │
     ▼
[1] Tokenize (分词): [1234, 5678, 9, ...] ◄── 询问分词器
     │
     ▼
[2] 缓存查询: "我们以前见过 [1234, 5678] 吗？" ◄── 查询 RadixCache
     ├─ 是：发现 15 个已缓存的 token ──► 跳过这些 token 的 KV 计算
     └─ 否：缓存未命中                  ──► 计算所有 KV
     │
     ▼
[3] 锁定缓存: 增加引用计数（防止被驱逐）
     │
     ▼
[4] 分配存储: 在 KVPool 中为新 token 预留空间
     │
     ▼
[5] 创建 CacheInstruction: 打包缓存决策
     │
     ▼
[6] 前向传播 (Prefill 阶段):
     ├─ 对所有 token 进行分词和嵌入
     ├─ 针对每一层：
     │  ├─ 跳过已缓存 token 的 KV 计算（从池中重用）
     │  └─ 仅为新 token 计算 KV → 写入分配的槽位
     ├─ 计算最后一个 token 的 logits
     └─ 采样下一个 token
     │
     ▼
[7] 生成循环 (Decode 阶段):
     针对每个新 token：
     ├─ 创建 CacheInstruction: "追加此新 token"
     ├─ 前向传播 (1 个 token):
     │  ├─ 重用所有先前的 KV（现在已完全缓存）
     │  ├─ 为 1 个新 token 计算 KV
     │  └─ 获取 logits，采样
     └─ 追加到输出
     │
     ▼
[8] 完成请求:
     ├─ 将完整序列插入 RadixCache（用于未来的前缀匹配）
     ├─ 解锁（减少引用计数）
     └─ 将文本返回给客户端
```

---

## 关键组件

### 1. RadixCache (infer-scheduler/src/kv_cache/radix_cache.rs)

**用途：** 维护一个具有 O(1) LRU 驱逐机制的前缀共享 KV 缓存。

**核心操作：**

```rust
// 查询缓存中匹配的前缀
let (matched_indices, node_handle) = cache.match_prefix(&key);
// 返回：有多少个 token 已被缓存

// 锁定节点以防止在请求处理期间被驱逐
cache.inc_lock_ref(node_handle);

// 将完成的序列插入缓存
cache.insert(&key, value_indices);

// 解锁节点（可能变为可驱逐状态）
cache.dec_lock_ref(node_handle);

// 为新 token 分配存储空间
let indices = cache.alloc_tokens(num_tokens)?;

// 当内存满时，驱逐最近最少使用的条目
let freed = cache.evict(num_tokens_needed);
```

**内部结构：**

```
基数树 (前缀匹配):
├─ [根节点]
│  ├─ [1, 2, 3] → 节点 (引用计数: 1, 值: [0, 1, 2])
│  │  └─ [4, 5] → 节点 (引用计数: 0, 值: [3, 4])
│  └─ [1, 2, 6] → 节点 (引用计数: 1, 值: [5, 6])
│
LRU 列表 (驱逐顺序):
  [头部] → 节点([4,5]) → 节点([1,2,6]) → [尾部]
  ↑ 最旧 (先驱逐)              最新 ↑
```

### 2. ConcurrentRadixCache (infer-scheduler/src/kv_cache/concurrent_cache.rs)

**用途：** 使用 `RwLock` 封装 `RadixCache`，使其具备线程安全性。

**关键设计决策：**

| 操作 | 锁类型 | 为什么 |
|-----------|-----------|-----|
| `match_prefix` | 写锁 | 需要触碰 LRU（更新访问时间） |
| `get_stats` | 读锁 | 仅查询，无修改 |
| `insert` | 写锁 | 修改树结构 |
| `inc_lock_ref` | 写锁 | 更新节点状态 |
| `prepare_request` | 写锁 (原子) | 一次性完成匹配 + 锁定 + 分配 |

**线程安全：** 生成计数器（Generation counters）用于检测过期的句柄。

```rust
// 句柄包含生成 ID
// 如果缓存被清空，生成 ID 增加
// 旧句柄失效，防止“释放后使用”
let handle = cache.match_prefix(&key)?;
// ... 缓存被清空 ...
cache.inc_lock_ref(handle)?;  // ❌ 错误：句柄已过期！
```

### 3. ConcurrentInferenceEngine (infer-scheduler/src/concurrent_engine.rs)

**用途：** 协调多个并发请求，管理缓存决策。

**关键特性：**

```rust
// 将请求作为异步任务派生（与其他请求并发执行）
let response = engine.process_request(request).await?;

// 以真正的并行方式处理批次（任务并发派生）
let responses = engine.process_batch_parallel().await;

// 获取统计信息
let stats = engine.engine_stats().await;
println!("已完成: {}, 缓存命中率: {:.1}%",
    stats.completed_requests,
    hit_rate);
```

**内部协调：**

```
阶段 1: 分词 (快)
    ├─ 释放模型锁
    └─ 允许其他请求并行

阶段 2: 查询缓存 (RwLock - 并发读取)
    ├─ 多个请求可以同时查询
    └─ 缓存决定哪些内容已经计算过

阶段 3: 分配槽位 (RwLock 写锁)
    ├─ 为新 token 预留内存
    └─ 与其他分配操作串行化

阶段 4: 前向传播 (模型 Mutex - 串行)
    ├─ GPU 一次只能运行一个推理
    ├─ 但缓存查询是并行发生的！
    └─ 最大化每个 GPU 周期的有效工作量

阶段 5: 生成 Token (模型 Mutex)
    └─ 同阶段 4

阶段 6: 完成 (RwLock 写锁)
    ├─ 插入缓存供未来请求使用
    └─ 释放资源
```

### 4. 具备缓存感知的 Llama3 模型 (infer-worker/src/model/llama3.rs)

**用途：** 使用引擎发送的缓存指令执行前向传播。

**缓存感知方法：**

```rust
// 使用缓存指令执行完整前向传播
fn forward_with_cache(
    &mut self,
    tokens: &Tensor,
    instruction: &CacheInstruction,  // 重用/计算哪些内容
    kv_pool: &mut KVCachePool,
) -> Result<i32> { ... }

// Prefill 阶段：处理整个提示词
fn prefill_with_cache(...) -> Result<i32> { ... }

// Decode 阶段：一次处理一个 token
fn decode_with_cache(...) -> Result<i32> { ... }
```

**CacheInstruction 包含的内容：**

```rust
pub struct CacheInstruction {
    pub request_id: String,
    pub cached_indices: Vec<usize>,    // ← 从 KVPool 读取这些索引
    pub new_indices: Vec<usize>,       // ← 将结果写入 KVPool 的这些索引
    pub seq_start_pos: usize,          // ← 用于 RoPE 位置编码
    pub total_seq_len: usize,
}
```

---

## 使用模式

### 模式 1：基础服务器设置

```rust
use infer_engine::{ConcurrentInferenceEngine, ConcurrentEngineConfig};
use infer_engine::concurrent_zmq_server::{run_concurrent, ConcurrentServerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // 加载模型
    let model = Llama3::new(
        "/path/to/model",
        DeviceType::Cuda(0),
        false
    )?;

    // 配置引擎
    let engine_config = ConcurrentEngineConfig {
        max_batch_size: 32,
        max_queue_size: 256,
        max_concurrent_requests: 64,  // 关键参数！
        ..Default::default()
    };

    let engine = ConcurrentInferenceEngine::new(model, engine_config)?;

    // 启动高并发服务器
    let server_config = ConcurrentServerConfig::default();
    run_concurrent(engine, "tcp://0.0.0.0:5555", server_config).await?;

    Ok(())
}
```

### 模式 2：处理单个请求

```rust
// 在异步上下文中
let request = InferenceRequest {
    request_id: uuid::Uuid::new_v4().to_string(),
    prompt: "Translate: Hello".to_string(),
    max_tokens: 100,
    temperature: 0.7,
    ..Default::default()
};

let response = engine.process_request(request).await?;
println!("生成内容: {}", response.text.unwrap());
```

---

## 并发模型

### 锁分层结构

为了防止死锁，始终按以下顺序获取锁：

1. **RadixCache (RwLock 读/写锁)** ← 最高优先级
   └─ 快速操作，很快释放
2. **KVPool (RwLock 写锁)**
   └─ 分配/释放存储空间
3. **Model (Mutex)**
   └─ 耗时操作（前向传播）
      最后获取，持有时间最长

### 读写锁如何最大化并发

```
多个读取者 (缓存查询):
┌─────────────────────────────────────┐
│  请求 A: match_prefix()          │
│  请求 B: match_prefix()  ← 并发！
│  请求 C: match_prefix()  ← 同时读取
│  请求 D: get_stats()     ← 同样并发
└─────────────────────────────────────┘

排他性写入者 (缓存修改):
┌─────────────────────────────────────┐
│  请求 E: inc_lock_ref() ← 只有 E 持有锁
│  (其他请求等待)
└─────────────────────────────────────┘
```

**关键点：** 读者永远不会阻塞读者，这为读多写少的缓存查询提供了极高的吞吐量。

---

## 性能特性

### 内存使用量

典型配置（28 层，256 kv_dim，65536 max_tokens）：

- **K 缓存:** 28 * 65536 * 256 * 2 字节 = 233 MB
- **V 缓存:** 28 * 65536 * 256 * 2 字节 = 233 MB
- **元数据:** ~1 MB (树节点, 指针)
- **总计:** ~467 MB（可通过配置调整）

---

## 故障排除

### 问题：缓存命中率低

**可能原因：**
1. **提示词过于多样化** - 每个请求使用完全不同的提示词。
2. **请求到达过快** - 第一个请求尚未完成（未缓存）第二个请求就到达了。
3. **缓存驱逐** - 旧提示词在重用前被移除。
   - *解决方案：* 增加配置中的 `max_cache_tokens`。

### 问题：内存溢出 (OOM)

**可能原因：**
1. **最大缓存 token 数设置过高** - 减少 `max_cache_tokens`。
2. **并发请求无限制增长** - 降低 `max_concurrent_requests`。
3. **驱逐机制失效** - 检查是否因为节点被锁定导致无法驱逐。确保每个请求都调用了 `dec_lock_ref`。

### 问题：高锁竞争

**症状：** CPU 使用率高但 GPU 利用率低。
**解决方案：** 减少 `max_concurrent_requests`。虽然允许更多请求并行有好处，但过多的竞争会适得其反。

---

## 最佳实践

1. **批量处理相似请求：** 具有相同前缀的请求（如翻译任务、长文档问答）将显著受益于缓存。
2. **针对硬件调优：** 根据 GPU 显存大小调整 `max_cache_tokens` 和 `max_concurrent_requests`。
3. **妥善释放资源：** 在 `inc_lock_ref` 后务必对应调用 `dec_lock_ref`，否则会导致内存泄漏（锁定状态的 token 无法被驱逐）。
4. **监控关键指标：** 定期检查吞吐量 (tokens/sec) 和缓存命中率。

---

## 总结

RustInfer 的高并发系统通过以下方式实现了 2-3 倍的吞吐量提升：

1. **RadixAttention 前缀共享** - 缓存公共提示词前缀的计算结果。
2. **基于 RwLock 的细粒度锁** - 允许并发进行缓存查询。
3. **信号量限制** - 防止资源枯竭，同时保持 GPU 繁忙。
4. **引擎-工作器模式** - 将缓存管理（CPU）与模型执行（GPU）分离。

结果：**处理多个请求的速度比单独处理快得多**，具有智能的缓存重用和极低的锁竞争。