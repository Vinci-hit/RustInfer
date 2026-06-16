# infer-scheduler Crate 代码审查报告

## 组件概述

### 架构理解
`infer-scheduler` 是生产级连续批处理调度器，六边形 DDD + typestate 生命周期。核心设计：单 tokio `select!` 事件循环（`event_loop.rs`），MsgPack 反序列化 offload 到后台 task（`engine.rs:decode_worker_output`）；Worker 拥有物理 KV，调度器侧用 `RadixTree`（前缀复用索引）+ `KvBudget`（容量闸门）。KV 压力事件驱动（`AllocFailed` → round0 LRU 驱逐 ≤5%，round1 抢占）。控制平面由独立 std 线程 `router_thread` 拥有 ROUTER socket，经 `PendingCalls` 做 RPC 关联。typestate 生命周期 `Queued/Prefilling/Decoding/Finished`。

### 整体质量评价
质量很高，是三个 crate 中工程化最成熟的：typestate 让非法状态转换编译期失败、`KvBudget` 单一不变量 + reserve-on-report + pending_prefill 自愈、RadixTree owners 引用计数 LRU 准入规则清晰、注释解释了大量非显然决策、测试覆盖扎实（KvBudget/RadixTree/ingestion/control_event 都有单元测试）。主要风险集中在 **RadixTree `split_edge` 的 owner 划分逻辑过度复杂且存在正确性隐患**、**worker_lost / 单 worker 容错较弱**、**几处性能可优化点（O(n) 线性扫描、O(子树) 递归计数在热路径）**、以及 **KvBudget reserve 与 RadixTree release 的守恒一致性依赖调用方**。

### 问题数量统计
| 级别 | 数量 |
|------|------|
| Critical | 0 |
| High | 5 |
| Medium | 9 |
| Low | 7 |
| **总计** | **21** |

---

## High

### H1. RadixTree `split_edge` owner 划分逻辑自相矛盾、注释与代码不符，存在 pin 泄漏/误评风险
- **文件:行号**：`infrastructure/kv_cache/radix_tree.rs:488-593`
- **类别**：逻辑
- **描述**：`split_edge` 计算 `suffix_owners = owners_past ∪ suffix_transitive_owners`，但 `prefix_owners = self.nodes[node].owners.clone()`（原封不动）。注释（`:504-540`）反复推翻自己（"Simpler: prefix_owners = unchanged"），最终 prefix 保留**全部**原 owners。问题：若某 owner 的 `tip` 恰好停在 split 点 `pos` 且其链不经过 suffix，它本不该再 pin prefix 之后的内容——但因 prefix 保留全量 owners，该节点（prefix）即使所有真实 owner 都已 finish 也可能因残留 owner 永不进入 LRU，造成**前缀 KV 永久不可驱逐**（pin 泄漏）。反之若划分把某 owner 漏给 suffix，则 suffix 可能被提前驱逐而 alias 活跃序列。当前测试未覆盖"mid-edge 分叉 + 多 owner + 部分 finish"的组合路径。
- **修复建议**：用清晰不变量重写：owner 留在某节点 iff 其链覆盖该节点的至少一个 token。即按每个 owner 的 `(leaf, pos)` 精确判定 prefix/suffix 归属，而非"原样克隆 + 事后补 suffix"。补充组合测试（≥2 owner、tip 分布在 pos 两侧 + 顺序 finish）。

### H2. `worker_lost` / 单 worker 致命错误直接 Terminate 整个调度器，无重连/降级
- **文件:行号**：`application/control_event.rs:319-346`、`engine.rs:335-347`；`router_thread.rs:152-166`（CallOne 发送失败仅靠超时）
- **类别**：逻辑（容错）
- **描述**：单 rank 部署下 worker 心跳超时 → `WorkerLost` → `ControlOutcome::Terminate` → 事件循环退出，整个 scheduler 进程结束。任何瞬时网络抖动 / worker GC stall 超过 liveness 阈值都会击垮调度器，所有排队请求一并失败。对"生产级"而言缺少重连窗口 / 优雅降级。
- **修复建议**：区分"可恢复掉线"（进入 draining/重连等待，保留 waiting 队列）与"不可恢复"；至少给 worker 一个重注册宽限期再 Terminate。`CallOne` 发送失败应主动 resolve pending 为错误而非纯靠 deadline（见 `router_thread.rs:158-165` 注释自承认"Acceptable failure mode"）。

### H3. `lru_total_indices` 在 AllocFailed 热路径上做全树递归 O(N)
- **文件:行号**：`radix_tree.rs:385-392, 619-635`；调用点 `control_event.rs:175`
- **类别**：性能
- **描述**：每次 `AllocFailed round=0` 都调用 `lru_total_indices()`，它对 root 的每个子树做 `reclaimable_subtree_total` 完整递归（遍历整棵树所有节点）。在大前缀缓存 + 高 KV 压力（恰好是 AllocFailed 频发场景）下，这是 O(节点总数) 的扫描，且与驱逐本身串行在单事件循环上，会放大压力期的尾延迟。
- **修复建议**：维护一个增量计数器（节点进/出"可回收"状态时 ±len），或缓存上次结果并按脏标记失效；至少在 `evict_collect_at_least` 已能返回实际数量时，避免先全量算 total 再驱逐的双重遍历。

### H4. KvBudget 守恒依赖调用方成对操作，存在 drift 风险且只在 debug 断言
- **文件:行号**：`domain/kv_budget.rs:131-160`；`control_event.rs:180`（release 与 send 分离）；`output_fns.rs:192-218`（`feed_radix_assigned_indices` 入参带 budget 但 `let _ = budget;` 完全不更新 outstanding）
- **类别**：逻辑
- **描述**：`outstanding` 的 `+=` 语义注释说"worker 报告 assigned_indices 时 +len"，但 `feed_radix_assigned_indices` 拿到 `budget` 后 `let _ = budget;`——**根本没有在 report 时 reserve**。真正的 `try_reserve` 调用点不在此处，意味着 outstanding 的加减分散在多处且不对称：release 在 `control_event.rs:180` 驱逐时减、reserve 在别处加。一旦某条路径（如抢占 `preempt_to_queued` 释放的 KV）漏减/漏加，`outstanding` 与 worker 真实占用 drift，且 `release` 的下溢只在 `debug_assert`（release build 静默 saturating）。这类守恒 bug 在 release 下无声发生，最终表现为"明明有空间却一直 AllocFailed"或"over-commit 崩 worker"。
- **修复建议**：把 KvBudget 的 +/- 收敛到**唯一**入口（report 路径加、free/preempt 路径减），并在每次 AllocFailed 时用 worker 上报的真实占用做一次 reconcile（校正 drift）。下溢应升级为可观测的 error metric，而非仅 debug_assert。

### H5. RadixTree 节点 `Vec<Node>` 逻辑删除永不回收，长跑内存单调增长
- **文件:行号**：`radix_tree.rs:419-425`（逻辑删除不 compact）、注释 `:420-421`
- **类别**：性能/逻辑
- **描述**：`evict_collect_at_least` 驱逐节点时只 `clear()` 内容、不从 `self.nodes` 移除（为保持 NodeId 稳定）。长时间运行下 `nodes` 向量只增不减，墓碑节点累积；同时 `lru.generations` HashMap 也随 NodeId 增长。token_count/lru_total 等遍历 `nodes` 的操作成本随墓碑线性上升。
- **修复建议**：引入空闲 NodeId free-list 复用墓碑槽位（分配新节点优先从 free-list 取），或周期性 compact + 重映射 ChainTip/lru。至少回收 `lru.generations` 中已删除节点的条目。

---

## Medium

### M1. `scheduled_new_kv_tokens` 对每个 segment 做 O(hints) 线性 find，整体 O(n·m)
- **文件:行号**：`application/planning.rs:199-212`；`batch_builder.rs:172-175` 同样 `prefix_hints.iter().find`
- **类别**：性能
- **描述**：两处都对 `current_prefix_hints` 做线性 `find`。虽然单批 size 有界，但每迭代都重复，且 `build_prefill_cmd` 内对每个 seq 再 find 一次。
- **修复建议**：用 `HashMap<RequestId, &[GlobalIndex]>` 索引一次，O(1) 查；或在 `execute_plan` 阶段把 hint 直接挂到 chunk_sizes 同一结构里。

### M2. 全 prompt 前缀命中时丢弃缓存、强制重算（功能缺口）
- **文件:行号**：`planning.rs:149-156`
- **类别**：逻辑/性能
- **描述**：当 `matched_indices.len() >= input_ids.len()`（整段 prompt 命中缓存）时，代码 `mark_finished_chain` 撤销 pin 并清空 matched，**整段重新 prefill**。注释承认这是因为"没有 cached logits / no-write prefill 路径"。这放弃了最有价值的缓存命中（完全相同的 prompt 重复请求收益为零），是显著性能缺口。
- **修复建议**：实现"保留最后一个 token 重算"策略（vLLM 做法：命中 N-1 个，仅重算最后 1 token 以拿到 logits），把全命中的浪费从 O(prompt) 降到 O(1)。

### M3. `collect_failed_sequence_ids` 每次 clone + sort + dedup，且对全 running 集合扩展
- **文件:行号**：`control_event.rs:353-368`
- **类别**：性能
- **描述**：`err.sequence_ids.clone()` 后 `extend(running_sequence_ids())` 再 sort/dedup。`running_sequence_ids()` 本身（在 table 中）可能是一次全表收集。非致命错误的常见路径也走这套。
- **修复建议**：非 fatal 且 sequence_ids 非空时直接用给定列表，跳过全 running 收集；fatal 路径才全量。

### M4. `RequestId` 用 UUID v4，HashMap key 为 16 字节随机；热路径多次 `.clone()` + 字符串化日志
- **文件:行号**：`lifecycle.rs:37-53`；`engine.rs:185,340`、`planning.rs:119,167,172`、`ingestion.rs:174` 等大量 `request_id.clone()`
- **类别**：性能
- **描述**：UUID 选型对安全/去重有正当理由（注释说明），但作为 HashMap key + 频繁 clone（虽 Copy，16B 拷贝 + 日志 `%request_id` 触发 36 字符格式化）在高 QPS 下有非零成本。`Display` 在 debug 日志中频繁触发。
- **修复建议**：日志用 `tracing` 的惰性字段（已部分如此）；考虑内部再叠一层紧凑 u64 索引做 HashMap key，UUID 仅留作 external 关联。属优化非缺陷。

### M5. `process_llm_step_decoded` 流式 chunk 串行 await 逐条发送
- **文件:行号**：`output_fns.rs:279-281`
- **类别**：性能
- **描述**：收集 `token_chunks` 后逐条 `frontend.send_stream_chunk(...).await`。虽然底层是 unbounded channel send（不会真阻塞），但在 await 点逐个让出，且每条都 `ClientId::new(...to_vec())` 重新分配 identity（见 M6）。
- **修复建议**：批量 send（transport 提供 `send_many`），或减少 await 粒度。

### M6. `ClientId::new(client_id.as_bytes().to_vec())` 到处重复分配 identity 字节
- **文件:行号**：`output_fns.rs:77,97,121,332`；`zmq_transport.rs:199,209`
- **类别**：性能
- **描述**：每次发送响应都把 client identity `to_vec()` 克隆一份新 `ClientId`。流式高频路径累计大量小分配。
- **修复建议**：`ClientId` 内部用 `Arc<[u8]>` 或 `bytes::Bytes`，clone 变浅拷贝。

### M7. 单帧/半帧防御：control 与 frontend 的 recv 假设有 payload
- **文件:行号**：`router_thread.rs:228-241`（用指针相等判 single-frame，技巧脆弱）；`zmq_transport.rs:127-129`（`recv_bytes(0)` 阻塞收 delimiter/data，无 rcvmore 校验）
- **类别**：逻辑/健壮性
- **描述**：`router_thread` 用 `std::ptr::eq(last.as_ptr(), identity.as_ptr())` 判断是否只收到单帧——依赖 `last = identity.clone()` 的指针，逻辑正确但极其隐晦（克隆后指针不同，靠的是没进入 while 循环时 last 仍 = clone 的指针……实际 `last` 是 clone 出来的新分配，指针**永远**不等于 identity，该分支几乎走不到，real single-frame 不会被检测）。`zmq_transport.rs:127` 在 ROUTER 多帧未用 `get_rcvmore` 校验，畸形帧会错位解析。
- **修复建议**：统一用 `get_rcvmore()` 循环收齐所有帧再判定帧数；移除 `ptr::eq` 技巧。

### M8. metrics summary task / decode task 等后台 spawn 无 JoinHandle、无关停信号
- **文件:行号**：`event_loop.rs:33-53`（metrics 5s task，无限 loop 无退出）；`engine.rs:172`（decode task）
- **类别**：逻辑（资源/生命周期）
- **描述**：metrics summary task 是 `loop { ticker.tick().await }` 无退出条件，引擎退出后仍悬挂（依赖 `metrics_for_summary` Arc 与进程一起死）。decode task 有 channel 关闭退出，较好。规范上后台 task 应可优雅 join/取消。
- **修复建议**：给后台 task 传 `CancellationToken` 或在主循环退出时 abort 其 JoinHandle。

### M9. `RejectReason::as_message` 每次 `format!` 分配；拒绝路径无 metrics 计数
- **文件:行号**：`ingestion.rs:74-94`；`engine.rs:198-208`
- **类别**：规范/可观测性
- **描述**：拒绝原因每次 `to_string()/format!`，且 `engine.handle_new_request` 的 Rejected 分支只打日志，没有 `metrics.record_reject()` 类计数，生产无法统计拒绝率/原因分布。
- **修复建议**：拒绝按 reason 打点 counter；消息惰性构造。

---

## Low

- **L1** `engine.rs:130-137` `KvBudget::new(... .unwrap_or(u32::MAX))`：当 worker 未上报 `max_total_kv_tokens` 时容量退化为 `u32::MAX`，等于"无限预算"，AllocFailed 才兜底——应至少 warn。
- **L2** `kv_budget.rs:152-160` `release` 下溢仅 `debug_assert` + saturating，release build 静默（见 H4，单列规范项）。
- **L3** `radix_tree.rs:54` `EDGE_SPLIT_THRESHOLD=32` 魔法数字（有注释"tunable"但未配置化）。
- **L4** `control_event.rs:171` `let five_pct = total / 20;` 5% 阈值硬编码，注释提及但未提为 const / 配置。
- **L5** `radix_tree.rs:216-220` `debug_assert_eq!` global index drift 检查只在 debug；release 下 KV index 错配静默写入会导致错误 token 复用——建议升级为可观测告警。
- **L6** `ingestion.rs:144` `self.next_sequence_id += 1;` u64 自增无 wrap 处理（实际不可达，属注释完备性）。
- **L7** `event_loop.rs:40` `let elapsed = 5.0;` 硬编码假定 tick 精确 5s，长 GC 停顿下吞吐统计会偏高；应实测两次 snapshot 间隔。

---

## 总结
共深度审查 14+ 个 `.rs` 文件（含 42KB 的 radix_tree.rs），发现 **21 个问题**：Critical 0 / High 5 / Medium 9 / Low 7。
该 crate 工程质量最高，无致命缺陷。最优先：**H1（split_edge owner 划分重写 + 补组合测试，正确性风险最高）→ H4（KvBudget 守恒收敛到唯一入口 + reconcile）→ H2（单 worker 容错/重连）→ H3/H5（RadixTree 热路径 O(N) 与内存单调增长）**。
