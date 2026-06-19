# RustInfer 重构进度追踪

## 项目概览

**项目名称**: RustInfer - 高性能异构推理框架
**核心 Crate**: `infer-scheduler`、`infer-worker`
**架构特点**:
- 六边形 DDD 架构
- Session 状态机
- RequestTable 使用 SlotMap + Bucket 管理
- Worker 采用 ABC buffer 三级流水
- GlobalKvAllocator 支持 prefix caching

---

## 重构方案总览

共 18 项重构任务，截至 2026 年 6 月 19 日：
- ✅ **已完成**: 10 项 (55.6%)
- 🔄 **进行中**: 0 项 (0%)
- ⏳ **待实施**: 6 项 (33.3%)
- ❓ **需补充**: 2 项 (11.1%)

---

## ✅ 已完成的重构任务

### #1: 提取 decode 公共代码
**状态**: 已完成
**实施细节**:
- 新建 `application/decode_common.rs`，集中 `DecodePrep`、`DecodeInputs` 类型及 `build_decode_inputs`、`build_a_append`、`fail_decode_seqs`、`send_step_error` 辅助函数（此前在 `decode_engine.rs` 与 `worker_scheduler.rs` 中重复实现）。
- 删除 `worker_scheduler.rs` 中的死代码：`run_decode_step`、`prepare_decode_step`、`commit_decode_results` 及其重复的类型/辅助函数（已被 `DecodeEngine` 完全替代，全 crate 无调用方）。
- `decode_engine.rs` 改为引用 `decode_common`，仅保留引擎专属的 `prepare_step`/`commit_results` 方法和 `trace_decode_commit`。
- `worker_scheduler.rs` 仅保留 prefill 路径（`handle_prefill`、`PrefillCtx`、`SegmentPlan`、`assigned_runs`），`send_step_error` 从 `decode_common` 引入。
- 顺带补全 #16 遗留缺陷：`serve_loop.rs` 引用了未定义的 `MAX_CONSECUTIVE_PREFILL_ROUNDS`（提交 1cd848c 引入但未定义常量，导致 `--features cuda` 构建失败），现已定义为 `usize = 16`。
- `cargo check -p infer-worker --features cuda` 通过，无告警。

### #2: ingestion prompt 校验
**状态**: 已完成
**说明**: `validate()` 函数中已包含 ingestion prompt 校验逻辑，无需额外修改。

### #3: KV 一致性校验
**状态**: 已完成
**实施细节**:
- 在心跳结构中新增 KV 统计字段
- `KvBudget` 新增 `force_set_outstanding` 方法
- `control_fns` 实现漂移检测与自动校准逻辑
- 顺便修复了 `serve_loop.rs` 末尾的损坏代码

### #9: ActiveSeq 预分配优化
**状态**: 已完成
**实施细节**: `ActiveSeq::new` 构造函数预分配 `block_table` 容量，减少动态扩容开销。

### #12: GlobalKvAllocator bug 修复
**状态**: 已完成
**实施细节**: 修复 `total_free()` 漏计 `released` 列表的 bug，确保空闲内存统计准确。

### #15: poll_next_event 逻辑简化
**状态**: 已完成
**实施细节**: 简化 select! 逻辑，减少代码冗余，提升可维护性。

### #5: WaitingQueue 懒删除优化
**状态**: 已完成
**实施细节**:
- `WaitingQueue` 内部存储改为 `VecDeque<Option<InferenceSession<Queued>>>`，新增 `live` 计数。
- `remove()` 改为惰性删除：取出 session 后留下 `None` 墓碑，避免 `VecDeque::remove` 的 O(n) 尾部 memmove（高 QPS 取消路径热点）。
- `pop_front`/`front`/`iter`/`total_tokens` 跳过墓碑；`len`/`is_empty` 走 O(1) `live` 计数。
- 墓碑数超过阈值（`queue.len() > 2*live + 8`）时 `compact()` 批量回收，摊还 O(1)，存活顺序不变。
- 新增 5 个单测覆盖墓碑跳过、缺失 ID、front/total_tokens、churn 压缩、优先级保序；`cargo test -p infer-scheduler` 134 全过。
- 顺带补全 #3 遗留缺陷：`control_fns.rs` 测试构造 `WorkerHeartbeat` 缺 `kv_outstanding`/`kv_total_free`/`kv_released_pending` 三字段（编译失败），补 `None`。

### #6: prefill 零拷贝优化
**状态**: 已完成（GPU 验证）
**实施细节**:
- `handle_prefill` 构建 step 时不再克隆 `input_ids` 与 `block_table`：直接 move 进 `SeqStep`（forward 仅借用 `&[SeqStep]`）。
- `SegmentPlan` 新增 `step_idx: Option<usize>` 记录该段在 `steps` 中的下标；forward 返回后从 `steps[step_idx]` 用 `mem::take` 把 `input_ids`/`block_table` 移回 `token_ids`/`full_block_table`，供 `assigned_runs` 与 commit 阶段消费。
- 消除每个 prefill 段 2 次堆拷贝（`input_ids` 为整段 prompt，长 prompt 下最显著），对应审查 H1。
- 行为不变：仅 move 替换 clone，数据与时序一致。
**GPU 验证**（GPU 7，Llama-3.2-1B，`rustinfer.bench.toml`）:
- 单条 / 长 prompt：贪心输出连贯正确。
- 24 路并发同 prompt 贪心：输出 100% 一致（确定性保持，多段 prefill 批正确）。
- 30 路混合长度并发：30/30 有效。
- 200 请求 conc=32 持续压测：200/200 有效，5s，worker 日志 0 error/panic。

### #16: 魔术数字提取
**状态**: 已完成
**实施细节**: 提取魔术数字为 `MAX_CONSECUTIVE_PREFILL_ROUNDS` 常量（`serve_loop.rs`，值 16），提升代码可读性。
**备注**: 进度文档原记为 `PREFILL_ROUNDS_LIMIT`，但实际代码引用名为 `MAX_CONSECUTIVE_PREFILL_ROUNDS`；提交 1cd848c 引入引用却未定义常量，已于 #1 一并补全。

### #17: GlobalKvAllocator 初始化优化
**状态**: 已完成
**实施细节**: 优化 GlobalKvAllocator 的初始化逻辑，提升启动性能。

---

## ⏳ 待实施的重构任务

### #4+#10: ResourceContext 统一参数
**优先级**: 中
**说明**: 合并 `ResourceContext` 和 `DecodeContext` 的重复字段，统一参数传递接口。

### #8: WorkerState 持久化
**优先级**: 低
**说明**: 实现状态持久化与恢复机制，支持系统重启后的状态恢复。

### #7, #11, #13, #14, #18
**优先级**: 待评估
**说明**: 具体内容需进一步补充和明确。

---

## 技术债务清理记录

### 死代码识别（已清理 — #1）
- **文件**: `worker_scheduler.rs`
- **函数**: `run_decode_step`、`prepare_decode_step`、`commit_decode_results` 及重复的 `DecodePrep`/`DecodeInputs`/`build_decode_inputs`/`build_a_append`/`fail_decode_seqs`
- **原因**: 已被 `DecodeEngine` 完全替代，全 crate 无调用方
- **处理**: 已删除；公共类型/辅助函数提取至 `application/decode_common.rs`

---

## 备注

1. 本文档将持续更新，记录重构进展
2. 部分任务的具体细节需根据实际代码分析进一步补充
3. 建议优先实施高优先级任务（如 #6 prefill 零拷贝优化）

---

**最后更新**: 2026 年 6 月 19 日