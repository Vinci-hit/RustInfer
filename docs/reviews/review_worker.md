# infer-worker Crate 代码审查报告

## 组件概述

### 架构理解
`infer-worker` 是 GPU 推理运行时，六边形 DDD（domain ← infrastructure ← models ← application）。worker 拥有物理 KV 池（`GlobalKvAllocator` bump+sorted free-list）。`serve_loop` 是单线程事件循环：`zmq::poll` 多路复用 data PULL + control DEALER，prefill 先于 decode 执行（降 TTFT）。`worker_scheduler` 实现 `handle_prefill` / `run_decode_step`，KV 不足时经 `kv_relief::alloc_with_relief` 向调度器请求 round0/round1 救济。`ModelRunner` 编排 forward；`CudaGraphRunner` 管理 decode-only CUDA graph 捕获/回放（按 batch size padding）。CUDA 层经大量 `extern "C"` FFI 调 .cu kernel（matmul/flash_attn/scatter_kv 等）。

### 整体质量评价
功能完备、性能设计先进（CUDA graph、ABC compact decode、reserve-on-report KV、graph 友好的 copy-in/compute stream 分离）。但作为三个 crate 中**最不成熟**的：CUDA FFI 错误处理普遍缺失（kernel 返回 void、cuBLAS 返回码被丢）、热路径上大量 `block_table.clone()` 和 H2D 同步阻塞、`unsafe` 块的 SAFETY 注释多处与实际不符（`step_batch_eager` 的 prof 分支注释自承"D is Cuda here"但泛型实为任意 D）、`eprintln!` 满天飞（无结构化日志/级别）、DDD 依赖在 application 层穿透到 infrastructure ffi 细节。错误传播多用 `let _ = ...` 静默丢弃。

### 问题数量统计
| 级别 | 数量 |
|------|------|
| Critical | 2 |
| High | 8 |
| Medium | 10 |
| Low | 7 |
| **总计** | **27** |

---

## Critical

### C1. [cuda] CUDA kernel / cuBLAS 错误被系统性吞掉，GPU 故障静默产生错误结果
- **文件:行号**：`infrastructure/cuda/kernels/matmul.rs:10-72`（所有 FFI 声明返回 `void`，无返回码）、`matmul.rs:92-159`（unsafe 调用后直接 `Ok(())`，从不检查 `cudaGetLastError`）；同样模式遍布 `cuda/mod.rs` 各 kernel 包装
- **类别**：逻辑（正确性）
- **描述**：`sgemv_cu_fp32x4` / `gemm_cublaslt_bf16` / `kpack_gemv_cu` 等全部声明为返回 `void`，包装函数在 `unsafe { ... }` 后无条件 `Ok(())`。这意味着：(1) kernel launch 失败（非法配置、OOM、非法地址）不会被捕获；(2) cuBLASLt 内部错误码被忽略；(3) 后续 kernel 在错误的 stream 状态上继续执行，产出 NaN/垃圾 token 而无任何错误上报给调度器。对"生产级 + 高性能系统规范"这是最严重的缺陷——错误推理结果会被当作正常 token 返回给用户。
- **修复建议**：所有 kernel launch 后调用 `cudaPeekAtLastError()`（异步错误）并在关键边界 `cudaGetLastError()`；cuBLAS/cuBLASLt FFI 改为返回 `cublasStatus_t` 并检查。至少在每个 `step_batch` 末尾做一次 `cudaGetLastError` 校验，错误转 `OpError::Kernel` 上报，使 worker 能发 `StepError{fatal}`。

### C2. [application] `step_batch`/forward 路径不校验 `cudaGetLastError`，且 D2H 之前无显式错误同步——graph 回放失败静默
- **文件:行号**：`application/model_runner.rs:257-288`（forward + argmax，无错误检查）；`model_runner/cuda_decode.rs:267`（`config.launch(slot)?` 仅检查 launch API 返回，不检查 graph 内 kernel 运行时错误）、`:277`（`to_host_vec()` D2H 后未校验前序 kernel 是否出错）
- **类别**：逻辑
- **描述**：CUDA graph 回放（`config.launch`）只返回"提交成功"，图内 kernel 的运行时错误要到下一次同步点（`to_host_vec` 的 `cudaStreamSynchronize`）才暴露为 `cudaError`，但该同步的返回码（在 `download` 里）虽被检查，却归因到"D2H 失败"而非真实的 kernel 错误，且 `step_batch_with_graph` 的返回值 unwrap 路径不会区分。配合 C1，错误 token 会被静默接受。
- **修复建议**：forward 完成后、D2H 之前插入一次 `synchronize()` 并 `cudaGetLastError`（profiling 关闭时也要做，至少周期性 sticky-error 检查）；把 launch 后的首个同步错误明确归类为"compute error"而非 transport 错误。

---

## High

### A. application 层

### H1. [application] decode/prefill 热路径每序列每步 `block_table.clone()`，O(seq_len) 拷贝
- **文件:行号**：`worker_scheduler.rs:444`（`let mut bt = seq.block_table.clone();` 每个活跃序列每步克隆整张 block_table）、`:178-180,251-254`（prefill 拼 block_table）、`model_runner.rs:209-218`（`WsSeqStep` 把 input_ids/positions/block_table 全 clone）、`cuda_decode.rs:209,221-230,508,519-528`（padding 路径再 clone 一遍）
- **类别**：性能
- **描述**：block_table 随序列增长（decode 到第 N 步长度为 N），每步对每个活跃序列整表 `clone()` 是 O(batch · seq_len) 的纯 host 拷贝，在长序列大 batch 下成为可观的 host 开销，且 `seq.block_table.push(new_idx)`（`:531`）又额外维护一份。`SeqStep` → `WsSeqStep` 的全字段 clone 是纯适配开销。
- **修复建议**：block_table 用 `Arc<Vec<u32>>` 或在 workspace 内复用 staging buffer，传引用而非 clone；`SeqStep`/`WsSeqStep` 合并为同一类型消除适配 clone；decode step 用增量更新而非重建。

### H2. [application] `step_batch_eager` 的 profiling unsafe 分支 SAFETY 注释错误且泛型不安全
- **文件:行号**：`model_runner.rs:227-255, 268-285`
- **类别**：逻辑/规范（unsafe 安全）
- **描述**：`step_batch_eager` 是泛型 `<T, D, M>` 方法，但 prof 分支无条件用 `crate::infrastructure::cuda::ffi` 调 `cudaEventCreate`，SAFETY 注释自承"the generic D is Cuda here when this branch is taken at the call site"——这是**未经类型系统保证的假设**。若该泛型方法被 `D=Cpu` 实例化并设置 `RUSTINFER_PROFILE_GPU`，会对非 CUDA 上下文调 CUDA FFI（UB/崩溃）。当前靠"调用方约定"而非编译期约束，违反 unsafe 最小化与可证明安全原则。
- **修复建议**：把 CUDA profiling 代码移到 `impl ModelRunner<T, Cuda, M>`（`cuda_decode.rs` 那样），用类型而非运行时约定保证 D=Cuda；或用 `D::is_cuda()` 运行时门控并在非 Cuda 时彻底跳过 FFI。

### H3. [cuda] `MemoryPort::upload` 每次同步 `cudaStreamSynchronize`，prefill 构建计划时阻塞 GPU 流水
- **文件:行号**：`cuda/mod.rs:87-118`（`upload` 内 H2D 后立即 `cudaStreamSynchronize`）；对照 `upload_async`（`:120-145`，无同步）
- **类别**：性能
- **描述**：同步版 `upload` 在每次 H2D 后强制全流同步。`Tensor::from_host_slice`/`from_host_bytes`（`tensor.rs:69,103`）都走 `upload`（同步版），权重加载尚可接受，但若任何 per-step 路径误用同步 `upload`（而非 `upload_async`），就会在每步插入一次全 GPU 同步，扼杀 batching 吞吐。同步点过多是 GPU 推理的头号性能杀手。
- **修复建议**：审计所有 per-step 调用确保走 `upload_async` + 事件依赖（copy_in stream）；同步 `upload` 仅限初始化路径。给同步版加 `#[doc]`/命名警示（如 `upload_blocking`）防误用。

### H4. [application] `wait_for_relief` 收到 Shutdown 时 `std::process::exit(0)`，绕过所有析构与 KV 释放
- **文件:行号**：`kv_relief.rs:69-72`
- **类别**：逻辑（规范/资源）
- **描述**：在等待 KV 救济期间收到 `Shutdown`，直接 `std::process::exit(0)`——不返回到 serve loop、不走正常 drain、不释放显存/句柄、不发 ack。这在库代码深处硬退出进程是严重的设计气味，也使任何 RAII（cudaFree、socket close）失效，且无法被测试覆盖。
- **修复建议**：改为返回特殊标志（如 `Option<ReliefOutcome>` 含 `Shutdown` 变体）让调用栈逐层优雅退出 serve loop，由顶层统一退出。

### H5. [application] prefill 批量内 `drain_control` 在循环中反复调用，cancel 与正在构建的 prefill 存在 TOCTOU
- **文件:行号**：`serve_loop.rs:262-287`（每个 pending prefill 前都 `drain_control`）；`worker_scheduler.rs:135-150`（用 `prefilling.get` 判 stale，但 base_table 长度校验后才 skip）
- **类别**：逻辑（并发/竞态）
- **描述**：serve loop 在处理 pending_prefills 时，每条 prefill 前先 drain control（可能 cancel/preempt 掉某序列），但 `handle_prefill` 内部已基于进入时的 `cmd` 快照分配了 `base_indices`。若 cancel 命中本批某序列，该序列在 `handle_prefill` 内仍会被分配/写入（stale 检测只覆盖 base_table 长度不匹配，不覆盖"已被 cancel"），可能向已释放序列写 KV 或多分配 slot 后才丢弃。逻辑虽用 skip+unused_indices 兜底回收，但路径复杂、易漏。
- **修复建议**：`handle_prefill` 内分配前再次校验每个 segment 的 sequence_id 是否仍有效（未被本轮 cancel）；或把 cancel 集合传入，构建 steps 时直接剔除。

### H6. [application] `alloc_with_relief` 失败语义复杂、round 升级逻辑有难以验证的状态机
- **文件:行号**：`kv_relief.rs:106-177`
- **类别**：逻辑
- **描述**：`alloc_with_relief` 的循环混合了 `round`、`retried_after_round1_relief`、`shrink_to_active` 三个状态 + 多个 `continue`，分支组合多。例如：relief 成功但 `round==0` 时只置 `round=1` 不重试分配（靠下一轮循环），而 `shrink_to_active` 又可能在 relief 前后改变 `n`。这种隐式状态机缺单元测试（该文件无 `#[cfg(test)]`），KV 压力是最易出 corner case 的路径，无测试覆盖风险高。
- **修复建议**：拆成显式状态枚举或两段式函数（round0 尝试 / round1 尝试），补齐单元测试覆盖：relief 满足/部分满足/超时、shrink_to_active 边界、round 升级。

### H7. [domain] `GlobalKvAllocator::free` 每次 `sort_unstable` 整个 free 池，O(N log N) 在高频释放路径
- **文件:行号**：`domain/global_kv_alloc.rs:171-188`（`free` 内 `self.free.sort_unstable()`）；调用频度：每次序列完成/取消/抢占
- **类别**：性能
- **描述**：注释承认"O(N log N) per free()，但 freed 批次通常小"。问题是 `free.len()` 是**整个空闲池**（可达数万 token），不是 freed 批次大小——每次释放一个完成序列就对整池重排。高 QPS 短序列负载下（频繁完成→频繁 free），这是 O(总空闲量·log) 的反复全排序。`release`+`recycle` 模式（prefix caching disabled）较好（惰性），但 `free`（prefix caching enabled 路径）每次全排。
- **修复建议**：用有序结构（BTreeSet）或保持 free 池始终有序 + 归并插入（freed 批次先排序再 merge，O(N + k log k)）而非整池 sort。或像 `released` 一样惰性化 `free`。

### H8. [transport] 控制/数据响应发送全部 `let _ = ...` 丢弃错误，worker 与调度器可静默失联
- **文件:行号**：`worker_scheduler.rs:293,548`（`let _ = data.send_step_output(...)`）、`serve_loop.rs:376,415,431,441,488`（control send 全 `let _`）、各 StepError 发送 `let _`
- **类别**：逻辑（健壮性）
- **描述**：worker → scheduler 的所有输出（step output、StepError、ack、heartbeat）发送失败都被静默忽略。若 ZMQ 通道断裂或满，worker 会继续推理但调度器永远收不到 token/错误，表现为请求挂死直到超时；worker 自身无任何感知或重连。
- **修复建议**：发送失败应记录并触发 worker 侧的连接健康检查/退出（transport 断裂是致命的）；至少对 step output 发送失败做计数与告警。

---

## Medium

### M1. [application] `eprintln!` 作为主要日志手段，无级别/无结构化/直写 stderr
- **文件:行号**：`serve_loop.rs` 与 `worker_scheduler.rs`、`kv_relief.rs`、`cuda_decode.rs` 全篇大量 `eprintln!`
- **类别**：规范
- **描述**：worker 几乎全用 `eprintln!`（serve loop 内甚至在热路径打印 "cancelled seq" 等），无法按级别过滤、无结构化字段、高频时本身成为性能负担（stderr 加锁 + format）。scheduler/server 用的是 `tracing`，worker 不一致。
- **修复建议**：统一切到 `tracing`，热路径降为 `trace!`/`debug!` 并惰性求值。

### M2. [domain] `GlobalKvAllocator::new` 预填 `0..total` 一次性分配整个 Vec
- **文件:行号**：`global_kv_alloc.rs:75-86`
- **类别**：性能
- **描述**：大 KV 池（数十万 token）启动时 `for i in 0..total { free.push(i) }` 一次性物化整个索引 Vec，且后续每次 free 都参与排序（见 H7）。可用"bump 区间 + 显式空洞集"表示空闲，避免物化全部索引。
- **修复建议**：用区间表示初始全空闲（`head` 覆盖 `[0,total)` 连续区），只在产生空洞时记录，省去启动物化与全排序。

### M3. [application] CUDA graph 用环境变量 `RUSTINFER_DISABLE_GRAPH`/`PROFILE_GPU`/`FORCE_GEMM`/`TRACE_GRAPH` 在热路径反复 `std::env::var`
- **文件:行号**：`cuda_decode.rs:181,234,245,296,478`、`model_runner.rs:228`、`matmul.rs:126`
- **类别**：性能/规范
- **描述**：`std::env::var` 每次都查环境（加锁 + 分配 String），却在 `step_batch_with_graph`/`step_decode_abc_compact`/`matmul` 等每步热路径反复调用。
- **修复建议**：启动时读一次缓存为 `bool` 字段/`OnceLock`，热路径只读 bool。

### M4. [application] padding 路径为对齐 graph batch 反复构造 dummy `SeqStep` + clone block_table
- **文件:行号**：`cuda_decode.rs:206-230,506-528`
- **类别**：性能
- **描述**：每次 graph replay 都按 `padded_size - batch` 新建 dummy SeqStep（含 `pad_block_table.clone()` × 每行），再整体 `padded: Vec<SeqStep> = seqs.to_vec()`。padding 是高频 decode 路径，这些分配可预先复用。
- **修复建议**：预分配最大 padded 的 dummy steps 模板，复用；pad_block_table 用共享只读引用。

### M5. [domain] tensor `to_host_vec` 每次同步 D2H，compact decode 一步内做 5+ 次独立 D2H 同步
- **文件:行号**：`cuda_decode.rs:571,591-630`（counts + active_tokens + active_src_rows + finished_src_rows + finished_tokens 各一次 `to_host_vec`，各自 `cudaStreamSynchronize`）
- **类别**：性能
- **描述**：`step_decode_abc_compact` 末尾连续 5 次 `to_host_vec`，每次内部都 `cudaStreamSynchronize`（见 `tensor.rs` → `download`）。前 4 次的数据本可在一次同步后批量读取（第一次同步已确保 stream 完成）。多次冗余同步增加 CPU-GPU 往返。
- **修复建议**：合并为一次 D2H（把 counts/src_rows/tokens 放连续 buffer 一次拷回），或用 `cudaMemcpyAsync` 批量发起后只同步一次。

### M6. [application] `handle_prefill` 失败路径重复"free base_indices + remove prefilling + 发 StepError"样板，易漏
- **文件:行号**：`worker_scheduler.rs:107-122,205-227,326-343,363-387,405-434,475-497`
- **类别**：规范（重复代码/健壮性）
- **描述**：prefill 与 decode 的多个错误分支都手写"释放 KV → 从 active/prefilling 移除 → 发 StepError(RequestId(0))"，逻辑相似但每处略有差异（有的 free base_indices、有的 release block_table），极易漏掉某次释放导致 KV 泄漏。
- **修复建议**：抽 `fail_batch(seqs, kv, reason)` 统一回收 + 上报；用 RAII guard 保证 base_indices 在错误路径必被 free。

### M7. [transport] `RequestId(0)` 硬编码作为非关联 StepError 的 envelope id，魔法值散落
- **文件:行号**：`worker_scheduler.rs:72,119,223,332,375,417,485`
- **类别**：规范（魔法数字）
- **描述**：每处发 StepError 都裸写 `infer_protocol::control_envelope::RequestId(0)`，"0 = 非关联"语义未命名常量化。
- **修复建议**：用 `RequestId::NONE` 命名常量（scheduler 侧已有 `is_correlated`/`NONE` 概念，统一）。

### M8. [application] `num_blocks` 预算计算用整数 `(free as f64 * fraction) as usize`，无显存碎片/对齐安全边际审计
- **文件:行号**：`serve_loop.rs:63-84`
- **类别**：逻辑
- **描述**：KV 池块数从 `cudaMemGetInfo` 的 free 字节 × fraction 推导，`saturating_sub(1)` 仅留 1 块边际。未考虑 graph workspace、cuBLASLt workspace、activation 峰值显存——若 fraction 偏高，运行期峰值 OOM 风险（且 OOM 经 C1 路径还会被吞）。
- **修复建议**：扣除已知 workspace（graph_mem/workspace_mem，capacity 里有字段但全填 None）后再算；保留更保守边际或运行期监控。

### M9. [domain] `block_size != 1` 仅 warn 不拒绝，上层不变量会静默破坏
- **文件:行号**：`model_runner.rs:110-121`
- **类别**：逻辑
- **描述**：注释明确 `GlobalKvAllocator` 和 RadixTree handle 都依赖 `block_size == 1`，但代码只 `tracing::warn!` 继续运行。若误配 block_size>1，`block_table[seq][i]` 不再是 i-th token 全局索引，KV 复用/释放会错乱，但无硬失败。
- **修复建议**：在 worker-owned KV 路径强制 `block_size == 1`，否则返回 `OpError::Shape` 拒绝启动。

### M10. [models] safetensors/loader 与 build.rs（13.7KB）未在本次深读，存在加载期校验盲区（建议补审）
- **文件:行号**：`infrastructure/io/safetensors.rs`、`models/loader.rs`、`build.rs`
- **类别**：规范
- **描述**：模型权重加载正确性（dtype 转换、shape 校验、tied weight 共享）是正确性关键路径，本次聚焦运行时未深审。`from_host_bytes` 的 `bytes.len()==numel*SIZE` 校验是好的基础，但 loader 层的 dtype/shape 一致性需独立核查。
- **修复建议**：单独对 loader 做一次 dtype/shape/对齐校验审查。

---

## Low

- **L1** [cuda] `cuda/mod.rs:61` SAFETY 注释"cudaMalloc/cudaMemset 安全"过于笼统；`alloc_bytes` 用 `size.max(1)` 把 0 字节分配为 1，未文档化语义。
- **L2** [application] `serve_loop.rs:54` `(max_seq_len + block_size - 1) / block_size` 应用 `div_ceil`（标准库已有，`model_runner.rs:338` 已用 `div_ceil`，不一致）。
- **L3** [application] `serve_loop.rs:24-27` `cudaProfilerStart/Stop` 的 `extern "C"` 返回码 `u32` 被忽略（`unsafe { cudaProfilerStart(); }` 不检查）。
- **L4** [domain] `global_kv_alloc.rs:108` `outstanding = total - total_free()`，依赖 `total_free <= total` 不变量，若 free 误传入越界索引（release build 跳过校验）会下溢 panic/wrap。
- **L5** [application] `cuda_decode.rs:74,207,506` `scratch_block = num_blocks - 1` 反复硬算同一值，且依赖"最后一块保留"约定仅靠注释（`:46-48`），无运行期保护防止真实分配用到它。
- **L6** [application] `matmul.rs:75-160` 不校验 `output` 形状是否为 `[m, n]`，依赖调用方；越界由 kernel 内部决定（结合 C1 错误被吞更危险）。
- **L7** [transport] `drain_data` 对 `DiffusionBatch` 在 LLM worker 里回一个空 `DiffusionBatchOutput`（`serve_loop.rs:464-466`），静默吞掉模态不匹配的请求，应上报错误。

---

## 总结
共审查 application（serve_loop/worker_scheduler/model_runner/cuda_decode/kv_relief/cuda_graph_runner）、domain（global_kv_alloc/tensor）、infrastructure（cuda/mod/matmul/thread_stream）等核心文件，发现 **27 个问题**：Critical 2 / High 8 / Medium 10 / Low 7。

该 crate 是三者中风险最高的。最优先：**C1+C2（CUDA/cuBLAS 错误处理缺失——错误推理结果会被静默返回，必须先修）→ H8（输出发送失败静默致失联）→ H1/H3/H7（热路径 clone + 同步阻塞 + 全池排序，性能三大点）→ H2/H4（unsafe 泛型安全 + process::exit 硬退出）**。
