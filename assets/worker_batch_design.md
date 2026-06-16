# Worker Continuous Batching 设计文档

> 状态：历史设计稿。当前 worker 已重构为
> `crates/infer-worker/src/application/`、`domain/`、`infrastructure/`、`models/`
> 四层结构，并已采用 paged KV / CUDA Graph decode 路径。本文保留设计背景和
> 决策脉络，文件清单与部分 Phase 2/3 状态不代表当前源码布局。

## 1. 设计背景

RustInfer 原有架构中，`infer-scheduler` 直接调用 `infer-worker` 的 `model.generate()` 逐请求串行推理。这种方式下 GPU 利用率极低——每个请求独占 GPU 做完整的 prefill + decode 循环，其他请求排队等待。

生产环境需要 **continuous batching**：多个请求的 decode 步骤合并到同一个 GPU batch 中执行，一次 GEMM 调用处理 B 个请求，GPU 利用率与 B 成正比。

## 2. 核心设计决策

### 2.1 Worker 是独立进程

参考 vLLM 的多卡部署模型：每张 GPU 一个 Worker 进程，调度器通过 IPC 与多个 Worker 通信。当前只做单卡，但架构上为多卡预留。

```
调度器进程 ──ZMQ──→ Worker 进程 0 (GPU 0)
            ──ZMQ──→ Worker 进程 1 (GPU 1)   ← 未来
```

### 2.2 两线程模型: Server + Runner

Worker 内部只有两个线程，职责严格分离：

| 线程 | 职责 | 阻塞点 |
|------|------|--------|
| **Server** | 一切 CPU 操作：ZMQ 收发、KV cache 管理、EOS 判断、batch 组装、H2D copy | ZMQ recv / spin on output_ready |
| **Runner** | 纯 GPU 操作：forward pass、sampling。不解析任何业务逻辑 | spin on input_ready |

**为什么不用三线程（Prepare + Runner + Postprocess）？**

讨论过三线程方案，但发现：
- Server 的"读 output + 写 input"是快路径（~50μs），EOS/ZMQ 是慢路径
- 把慢路径放到 Runner 执行期间并行做，气泡只有快路径的时间
- 三线程多一次 channel 传递反而增加延迟

### 2.3 共享显存通信，不用队列

最初设计用 crossbeam channel 传递 Tensor 对象。后来发现：

**预分配固定大小 GPU buffer + 原子信号量** 更优：
- 零拷贝：Runner 直接从共享 buffer `slice_view` 出 GPU tensor
- 零分配：每步只是覆盖写同一块内存
- 零序列化：不需要构造/析构 Tensor 对象

```
Server 写 input buffer → store(input_ready) → Runner load(input_ready) → 跑 GPU
Runner 写 output buffer → store(output_ready) → Server load(output_ready) → 读结果
```

**安全性**：
- `input_ready == 0` 时 Server 才写 input（Runner 不碰）
- `output_ready > 0` 时 Server 才读 output（Runner 不碰）
- 所有数据写完后最后才 store ready（Release），对方 load（Acquire）后才读

### 2.4 Server 工作顺序最小化气泡

```
Runner GPU:    ████ step N ████                     ████ step N+1 ████
                              │output_ready         │input_ready
Server CPU:                   ▼                     ▲
                              [read output][write input]  ← 快路径 (~50μs)
                                                    │
                              [EOS判断][ZMQ发送]     │  ← 慢路径 (并行)
```

关键：**先把下一步 input 准备好发给 Runner，再慢慢做 EOS/ZMQ**。Runner 不用等 Server 做完所有事情。

### 2.5 调度器只发 Prefill，Decode 由 Worker 自循环

传统方案中调度器管 decode 循环——每步收结果、判断、再发指令。这引入了不必要的 IPC 往返延迟。

我们的做法：
- 调度器只发新请求的 prefill tokens
- Worker 内部自动 decode 循环，直到 EOS 或 max_tokens
- **每步结果通过 ZMQ_OUT 全量发回调度器（发完不管）**
- Worker 自己判 EOS 移除已结束的 seq
- 调度器收到结果后自行判断、释放 slot 等

两边独立判断，不冲突。

### 2.6 Batch 排列: Decode 在前，Prefill 在后

```
token_ids: [ d0, d1, ..., dN, p0_tok0, p0_tok1, ..., p1_tok0, ... ]
            |<-- decode --->|  |<----------- prefill ---------->|
```

理由：
- Decode 是稳态，形状固定（每个 seq 1 token），放前面利于 CUDA Graph
- Prefill 是新追加的变长数据，放后面不影响前面的布局
- 有新 prefill 时构造 `MixedBatch`，无 prefill 时构造 `DecodeOnly`

### 2.7 CUDA Graph 双路径

| batch_type | 场景 | 策略 |
|------------|------|------|
| `0` (DecodeOnly) | 全部 seq_len=1 | **Full CUDA Graph**：整个 forward capture 为一张图 |
| `1` (MixedBatch) | 含 prefill | **分段图**：attention 不入图（cu_seqlens 每次变），前后部分入图 |

```
DecodeOnly:  [entire forward] → 一张完整 graph, replay 极低开销

MixedBatch:  [Pre-Attn Graph] → [Attention 不入图] → [Post-Attn Graph]
             embedding/norm/QKV   flash_attn kernel    O proj/FFN/norm
```

## 3. Batch Forward 实现

### 3.1 现有 Op 的 Batch 能力

大部分 op 天然支持 batch——因为它们本质上是 `[M, N]` 的矩阵运算，M 从 1 变成 B 就是 batch：

| Op | 当前 shape | Batch shape | 需要改动 |
|----|-----------|-------------|---------|
| Embedding | `[1] → [1, dim]` | `[B] → [B, dim]` | ✅ 已支持 |
| RMSNorm | `[1, dim]` | `[B, dim]` | ✅ element-wise |
| Matmul | `[1, M] × [M, N]` | `[B, M] × [M, N]` | ✅ GEMM 天然 |
| SwiGLU | element-wise | element-wise | ✅ |
| RoPE | `pos=[1]` | `pos=[B]` | ✅ kernel 已支持 batch pos |
| **scatter_kv** | 写 1 行到 1 个 cache | 写 B 行到 B 个 cache | ❌ 需要新方法 |
| **FlashAttn** | 1 query attend 1 cache | B query attend B cache | ❌ 需要新方法 |
| **Sampler** | `[vocab] → 1 token` | `[B, vocab] → B tokens` | ⚠️ 需要循环 |

**只有 3 个 op 需要新增 batch 方法。**

### 3.2 scatter_kv_batch

```rust
/// 写 B 行到 B 个不同 cache 的各自 position
pub fn scatter_kv_batch(
    k_caches: &mut [&mut Tensor],  // B 个独立 K cache
    v_caches: &mut [&mut Tensor],  // B 个独立 V cache
    src_k: &Tensor,                // [B, kv_dim]
    src_v: &Tensor,                // [B, kv_dim]
    positions: &[i32],             // [B]
)
```

Phase 1：循环调用现有的 `scatter_kv`，每次处理一个 seq。
Phase 2（TODO）：写一个 batched CUDA kernel，一次 launch 处理 B 个 seq。

### 3.3 FlashAttn batch decode

```rust
/// B 个 query 各 attend 自己的 KV cache
pub fn forward_batch_decode(
    q: &Tensor,              // [B, num_heads * head_dim]
    k_caches: &[&Tensor],    // B 个独立 K cache
    v_caches: &[&Tensor],    // B 个独立 V cache
    kv_lens: &[i32],         // [B] 每个 seq 的 KV 长度
    output: &mut Tensor,     // [B, num_heads * head_dim]
)
```

Phase 1：循环 per-seq launch `flash_decoding_cu`。
Phase 2（TODO）：batched kernel 内部并行处理 B 个 seq。

### 3.4 forward_batch_decode

在 `Llama3` 上新增的方法，组装完整 batch decode 流程：

```
1. 收集 B 个 state.output_token → input_tokens[B]
2. embedding([B]) → x[B, dim]
3. for each layer:
   - rmsnorm(x[B, dim]) → rms_out[B, dim]
   - wqkv matmul → qkv[B, q_dim+2*kv_dim]      ← 一次 GEMM, M=B
   - split → q[B, q_dim], k[B, kv_dim], v[B, kv_dim]
   - rope(positions[B], q, k)                     ← 每行不同 pos
   - scatter_kv_batch(B 个 cache, k, v, positions) ← 写 B 行
   - flash_attn per-seq(q[i], cache_i, kv_len_i) ← 循环 B 次
   - wo matmul [B, dim]                           ← 一次 GEMM
   - residual + ffn_rmsnorm
   - gate_up matmul → swiglu → down matmul       ← 一次 GEMM
   - residual
4. final rmsnorm → cls matmul → logits[B, vocab]
5. per-seq sampler → tokens[B]
```

**BatchWorkspace**：预分配 `[max_batch_tokens, dim]` 等 buffer，避免每步分配。

### 3.5 KV Cache 组织

当前保持每个请求独立的 KV cache `[max_seq_len, kv_dim]`（在各自的 InferenceState 中）。Batch attention 时传入 B 个 cache 指针。

未来优化方向：PagedAttention，统一大 KV cache + 页表索引。

## 4. ZMQ 通信协议

### 4.1 调度器 → Worker (ZMQ_IN, PULL)

```rust
struct PrefillBatchCmd {
    input_ids: Vec<i32>,          // 所有请求 token 拼成一维
    q_start_loc: Vec<u32>,        // 每个请求的起始偏移
    num_computed_tokens: Vec<u32>, // chunked prefill 已处理数
    kv_slots: Vec<u32>,           // 调度器分配的 KV slot
    sampling_params: Vec<SamplingParams>,
    request_metas: Vec<RequestMeta>,
}
```

**请求数 = `q_start_loc.len()`**，Vec 自带长度不需要额外字段。

### 4.2 Worker → 调度器 (ZMQ_OUT, PUSH)

```rust
struct StepOutput {
    tokens: Vec<SeqToken>,   // 每步全量输出
}
struct SeqToken {
    request_id: String,
    token_id: i32,
    finished: bool,          // Worker 侧判断
}
```

**每步所有活跃 seq 的 token 都发，发完不管。** 调度器收到后自行决定。

## 5. 共享显存结构

```rust
struct SharedBuffers {
    // Input (Server 写, Runner 读)
    input_token_ids: Tensor,    // GPU, [max_batch_tokens]
    input_positions: Tensor,    // GPU, [max_batch_tokens]
    input_q_start_loc: Tensor,  // GPU, [max_seqs + 1]
    input_context_lens: Tensor, // GPU, [max_seqs]
    input_slot_indices: Tensor, // GPU, [max_seqs]

    // Output (Runner 写, Server 读)
    output_token_ids: Tensor,   // GPU, [max_seqs]

    // 同步信号 (CPU 原子变量)
    input_meta: InputMeta,      // ready, batch_type, num_decode_seqs, ...
    output_meta: OutputMeta,    // ready
}
```

`write_input_i32` / `read_output_i32`：通过 `cudaMemcpy` 直接操作 GPU buffer 的裸指针，绕过 Rust 的 `&mut` 要求（同步协议保证独占访问）。

## 6. 历史实现清单

> 以下清单对应早期 worker 原型，当前源码已迁移到 `application/serve_loop.rs`、
> `application/worker_scheduler.rs`、`application/model_runner*.rs`、
> `application/batch_workspace.rs`、`domain/batch.rs`、`domain/global_kv_alloc.rs`
> 以及 `infrastructure/transport/*`。

### 6.1 早期新增文件

```
crates/infer-worker/
├── src/
│   ├── worker/
│   │   ├── mod.rs              — 模块导出
│   │   ├── protocol.rs         — ZMQ 协议定义
│   │   ├── shared_buffers.rs   — 共享 GPU buffer + 原子同步
│   │   ├── server.rs           — WorkerServer (快/慢路径)
│   │   ├── runner.rs           — ModelRunner (真实模型 forward)
│   │   ├── runner_dummy.rs     — DummyRunner (CPU 测试用)
│   │   └── batch_workspace.rs  — BatchWorkspace 预分配
│   └── bin/
│       └── worker_main.rs      — rustinfer-worker 可执行文件入口
├── tests/
│   ├── test_worker.rs          — Worker 管线端到端测试
│   └── test_batch_forward.rs   — Batch vs 串行一致性测试
```

### 6.2 早期修改的现有文件

| 文件 | 变更 |
|------|------|
| `Cargo.toml` | `[[bin]]` + zmq/rmp-serde/clap/tracing 依赖 |
| `src/lib.rs` | `pub mod worker;` |
| `src/model/llm/llama3.rs` | `forward_prefill`/`forward_decoding` 改为 pub；新增 `forward_batch_decode`、`tokenizer()`、`config()` |
| `src/op/flash_gqa.rs` | 新增 `forward_batch_decode` 方法 |
| `src/op/scatter.rs` | 新增 `scatter_kv_batch` 方法 |
| `src/tensor/mod.rs` | 新增 `write_from_i32_host`、`read_i32_to_host`、`view_prefix` |

### 6.3 测试结果

| 测试 | 内容 | 结果 |
|------|------|------|
| `test_worker_pipeline_cpu` | DummyRunner, 单请求, max_tokens=3 | ✅ |
| `test_worker_multi_request_batch` | DummyRunner, 2 请求不同 max_tokens | ✅ |
| `test_worker_with_llama3` | Llama3-1B CPU, "1+1=", 对比 baseline 逐 token | ✅ 10 tokens 完全一致 |
| `test_batch_decode_matches_serial` | batch_decode(B=1) vs serial forward_decoding | ✅ 6 tokens 完全一致 |

```
Serial tokens: [17, 198, 17, 10, 17, 28]
Batch  tokens: [17, 198, 17, 10, 17, 28]
All 6 tokens match!
```

## 7. 后续优化方向

### Phase 2: 性能优化
- **scatter_kv_batch CUDA kernel**：一次 launch 写 B 行（当前循环 B 次 launch）
- **Batched FlashAttn kernel**：一次 kernel 处理 B 个 seq 的 decode attention
- **Batch sampler kernel**：batch argmax 不循环
- **CUDA Graph capture**：DecodeOnly 路径整体入图

### Phase 3: 功能完善
- **forward_batch_mixed**：decode + prefill 混合 batch（attention 逐 seq，其余合并）
- **PagedAttention**：统一 KV cache + 页表，替代当前独立 cache
- **Chunked prefill**：长 prompt 分段 prefill
- **多卡 (TP)**：tensor parallel across GPUs
