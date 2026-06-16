# RustInfer 代码审查总报告（server / scheduler / worker）

> 审查方式：按组件拆分为三个 opus 级深度审查任务，逐文件通读三大数据/控制路径，再归并去重、统一分级。
> 审查维度：① 性能 ② 逻辑（正确性 / 并发 / 容错）③ 规范（DDD / unsafe / 错误处理 / 命名 / 魔法数字）。
> 仅审查 `infer-server`、`infer-scheduler`、`infer-worker` 三个 crate。
> 分报告：`review_server.md`、`review_scheduler.md`、`review_worker.md`（同目录）。

---

## 一、总体结论

RustInfer 整体架构成熟、设计先进（六边形 DDD + typestate 生命周期 + 单事件循环 + worker 拥有 KV + RadixTree 前缀复用 + CUDA Graph + ABC compact decode）。三个 crate 的工程成熟度梯度明显：

| 维度 | infer-scheduler | infer-server | infer-worker |
|------|:---:|:---:|:---:|
| 工程成熟度 | ★★★★★ 最高 | ★★★★☆ | ★★★☆☆ 最低 |
| 测试覆盖 | 扎实（单元+集成） | 偏少 | 偏少（KV/graph 关键路径缺测） |
| 错误处理 | 完整 | 良好（个别 unwrap） | **薄弱（CUDA 错误系统性吞掉）** |
| 主要风险 | RadixTree split 正确性、单 worker 容错 | 流式 O(n²) decode、tokenizer clone | **CUDA 错误吞掉、热路径 clone+同步** |

**问题统计（去重后）：**

| 级别 | server | scheduler | worker | 合计 |
|------|:---:|:---:|:---:|:---:|
| Critical | 1 | 0 | 2 | **3** |
| High | 6 | 5 | 8 | **19** |
| Medium | 11 | 9 | 10 | **30** |
| Low | 8 | 7 | 7 | **22** |
| **小计** | 26 | 21 | 27 | **74** |

---

## 二、必须优先修复（Critical / 阻断生产）

### P0-1 ⚠️ [worker] CUDA / cuBLAS 错误被系统性吞掉 —— 错误推理结果静默返回给用户
> 对应 worker C1 + C2。**这是全项目最严重的问题。**
- 所有 kernel FFI（`matmul.rs:10-72` 等）声明返回 `void`；包装函数 `unsafe { ... }` 后无条件 `Ok(())`，从不调 `cudaGetLastError`/检查 cuBLAS 状态。
- CUDA graph 回放（`cuda_decode.rs:267`）只确认"提交成功"，图内 kernel 运行时错误归因错位到"D2H 失败"。
- 后果：kernel launch 失败 / OOM / 非法地址 / cuBLAS 错误 → 产出 NaN / 垃圾 token，被当作正常结果返回。**违反"高性能系统规范"中最基本的错误可观测性。**
- 修复：每个 kernel launch 后 `cudaPeekAtLastError`，关键边界 `cudaGetLastError`；cuBLAS FFI 返回 `cublasStatus_t` 并检查；`step_batch` 末尾强制一次错误校验，错误转 `OpError` 上报 `StepError{fatal}`。

### P0-2 ⚠️ [server] 流式增量解码 O(n²) —— 长输出延迟与 CPU 瓶颈
> 对应 server C1。
- `decoder.rs:55-91` 每个 token 都 `decode` 整个累积缓冲 + 全量 char-delta，长度 N 的输出总成本 O(N²)。
- 修复：改滑动窗口/前缀提交式增量解码（vLLM `detokenize_incrementally`），只 decode 尾部少量 token。

---

## 三、跨组件共性问题（归并后的主题）

这些问题在多个 crate 重复出现，建议作为**专项整改**统一治理：

### 主题 A：错误被 `let _ =` / `unwrap` 静默处理
- worker H8：所有 worker→scheduler 输出发送 `let _ =` 丢弃，通道断裂时静默失联（`worker_scheduler.rs:293,548`，`serve_loop.rs` control send 全 `let _`）。
- server H1：SSE 每 chunk `serde_json::to_string(...).unwrap()` 可 panic 断连（`streaming.rs` 多处）。
- server H2：`HeaderValue::from_str(&id).unwrap()`（`request_id.rs:22`）。
- 整改方向：网络/序列化失败路径一律显式处理 + 计数 + 告警；transport 断裂视为致命并触发健康检查。

### 主题 B：热路径不必要的内存拷贝
- worker H1：decode/prefill 每序列每步 `block_table.clone()`（O(batch·seq_len)）。
- server H5：每个流式请求 `tokenizer.clone()`（数十 MB 深拷贝）。
- server M6/M7/M8、worker M4：每 chunk / 每图 padding 反复 clone 小对象。
- scheduler M4/M6：`request_id.clone()`、`ClientId::new(...to_vec())` 散落。
- 整改方向：`Arc` 共享只读大对象（tokenizer / block_table / ClientId 内部 `Arc<[u8]>`）；workspace 复用 staging buffer。

### 主题 C：GPU/IO 同步点过多阻塞流水
- worker H3：`MemoryPort::upload` 每次同步 `cudaStreamSynchronize`。
- worker M5：compact decode 一步内 5+ 次独立 D2H 同步。
- server H6：ZMQ 线程 `recv_bytes(0)` 半帧阻塞整个事件循环。
- 整改方向：per-step 路径统一走 `upload_async` + 事件依赖；批量 D2H 一次同步；IO 收帧加超时/`rcvmore` 校验，绝不无界阻塞事件循环线程。

### 主题 D：魔法数字 / 阈值未常量化或配置化
- server M2（180s）、L7（多处）；scheduler L3（EDGE_SPLIT_THRESHOLD=32）、L4（5%）；worker M7（RequestId(0)）、M3（env var 热路径反复读）。
- 整改方向：提为带文档的 `const` 或纳入配置；env flag 启动时缓存为 bool。

### 主题 E：DDD 依赖方向 / 模块边界
- server H4：`chat`（领域）依赖 `api::openai::types`（最外层 DTO）；`StreamHandle`（基础设施）被 API 层直接消费。
- worker H2：泛型 `step_batch_eager` 的 CUDA profiling unsafe 分支靠"调用方约定"而非类型保证 D=Cuda。
- 整改方向：定义中立领域类型做转换；CUDA 专用代码移入 `impl ...<Cuda>`。

---

## 四、各组件 High 级问题清单

### infer-server（6）
- H1 SSE `unwrap` 可 panic 断连。
- H2 header `from_str().unwrap()` + 回写客户端 header 未清洗。
- H3 非流式 oneshot 无 deadline，channel 满时 pending 永久泄漏。
- H4 DDD 依赖方向倒置（chat 依赖 DTO、StreamHandle 泄漏到 API）。
- H5 每流式请求全量 clone tokenizer。
- H6 ZMQ 线程半帧 `recv_bytes(0)` 无界阻塞事件循环。

### infer-scheduler（5）
- H1 **RadixTree `split_edge` owner 划分逻辑自相矛盾**，存在 pin 泄漏（前缀 KV 永不驱逐）/误驱逐（alias 活跃序列）风险，缺组合测试。
- H2 单 worker 心跳超时直接 Terminate 整个调度器，无重连/降级。
- H3 `lru_total_indices` 在 AllocFailed 热路径做全树递归 O(N)。
- H4 KvBudget 守恒依赖调用方成对操作，drift 仅 debug_assert，release 静默。
- H5 RadixTree 墓碑节点永不回收，长跑内存单调增长。

### infer-worker（8）
- H1 decode/prefill 热路径 `block_table.clone()` O(batch·seq_len)。
- H2 泛型 profiling unsafe 分支 SAFETY 不成立（D 未必 Cuda）。
- H3 同步 `upload` 阻塞 GPU 流水。
- H4 `wait_for_relief` 收 Shutdown 直接 `std::process::exit(0)` 绕过析构。
- H5 prefill 批内 cancel 与正在构建的 prefill TOCTOU。
- H6 `alloc_with_relief` 状态机复杂且零单元测试。
- H7 `GlobalKvAllocator::free` 每次对整空闲池 `sort_unstable`。
- H8 worker→scheduler 输出发送失败静默，可致失联。

---

## 五、建议的整改路线（按 ROI 排序）

1. **第一阶段（正确性，阻断生产）**：worker C1/C2（CUDA 错误处理）→ scheduler H1（RadixTree split 重写 + 补测）→ server C1（增量解码）。
2. **第二阶段（稳定性/容错）**：worker H4/H8、server H1/H3/H6、scheduler H2/H4（容错 + 失联 + KV 守恒）。
3. **第三阶段（性能）**：主题 B（clone）+ 主题 C（同步点）+ worker H7 / scheduler H3·H5（数据结构复杂度）。
4. **第四阶段（规范）**：主题 D（魔法数字）+ 主题 E（DDD/unsafe）+ Medium/Low 项；worker 统一切到 `tracing`。
5. **补审建议**：worker `models/loader.rs` + `safetensors.rs` 的 dtype/shape/对齐校验；scheduler `engine_tests.rs` 之外的并发 corner case 测试；.cu kernel 的访存/bank-conflict 专项（本轮以 Rust 侧为主）。

---

## 六、亮点（值得保留与推广）

- scheduler：typestate 生命周期、KvBudget 单一不变量 + reserve-on-report 自愈、RadixTree owners 引用计数 LRU、后台 decode offload、扎实单元测试。
- worker：CUDA Graph 捕获/回放（H2D 排除在图外）、ABC compact decode、copy-in/compute stream 分离、reserve-on-report KV、确定性 free-list（TP/PP 友好）。
- server：流式取消语义（StreamHandle::Drop）、UTF-8 跨 token 增量解码、ZMQ 单线程 + wakeup PAIR 桥接、统一 OpenAI 错误映射。
