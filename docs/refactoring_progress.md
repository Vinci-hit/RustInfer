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

共 18 项重构任务，截至 2026 年 6 月 18 日：
- ✅ **已完成**: 7 项 (38.9%)
- 🔄 **进行中**: 1 项 (5.6%)
- ⏳ **待实施**: 8 项 (44.4%)
- ❓ **需补充**: 2 项 (11.1%)

---

## ✅ 已完成的重构任务

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

### #16: 魔术数字提取
**状态**: 已完成
**实施细节**: 提取魔术数字为 `PREFILL_ROUNDS_LIMIT` 常量，提升代码可读性。

### #17: GlobalKvAllocator 初始化优化
**状态**: 已完成
**实施细节**: 优化 GlobalKvAllocator 的初始化逻辑，提升启动性能。

---

## 🔄 进行中的重构任务

### #1: 提取 decode 公共代码
**状态**: 进行中
**当前进展**:
- 分析发现 `worker_scheduler.rs` 中的 `run_decode_step` 为死代码（已被 `DecodeEngine` 完全替代）
- 计划提取 `DecodePrep`/`DecodeInputs` 等类型至 `decode_common.rs`
- 删除死代码，更新 `decode_engine.rs` 引用

---

## ⏳ 待实施的重构任务

### #4+#10: ResourceContext 统一参数
**优先级**: 中
**说明**: 合并 `ResourceContext` 和 `DecodeContext` 的重复字段，统一参数传递接口。

### #5: WaitingQueue 懒删除优化
**优先级**: 中
**说明**: 实现延迟删除机制，避免频繁内存操作，提升性能。

### #6: prefill 零拷贝优化
**优先级**: 高
**说明**: 减少 prefill 阶段的数据拷贝开销，提升推理效率。

### #8: WorkerState 持久化
**优先级**: 低
**说明**: 实现状态持久化与恢复机制，支持系统重启后的状态恢复。

### #7, #11, #13, #14, #18
**优先级**: 待评估
**说明**: 具体内容需进一步补充和明确。

---

## 技术债务清理记录

### 死代码识别
- **文件**: `worker_scheduler.rs`
- **函数**: `run_decode_step`
- **原因**: 已被 `DecodeEngine` 完全替代
- **计划**: 删除死代码，提取公共类型

---

## 备注

1. 本文档将持续更新，记录重构进展
2. 部分任务的具体细节需根据实际代码分析进一步补充
3. 建议优先实施高优先级任务（如 #6 prefill 零拷贝优化）

---

**最后更新**: 2026 年 6 月 18 日