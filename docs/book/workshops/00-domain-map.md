# 第 0 次练习：领域边界与请求时序

状态：作者的请求链口述已完成第一轮源码审阅，并按作者要求由助手修正、润色为[引言](../02-introduction.md)。下文保留原文和反馈；作者的独立复述、领域图与源码核对尚未完成。

目标：找到你已经能解释的部分，以及需要查证的边界，为第一节确定范围。

场景：单卡、dense、BF16 模型的一条普通文本请求，从进入服务到正常生成结束。暂时关闭前缀复用与推测解码，TP=1。此次是代码理解练习，无需启动 GPU。

建议用时 25 分钟。允许留白和写“待确认”，不要求覆盖整个系统。

## A. 凭理解画边界：8 分钟

先不查源码，写出你认为参与该场景的 3–5 个候选领域或职责单元。名称由你决定。

对每个单元说明：负责什么决策、拥有什么状态、必须保证什么。

| 候选领域或职责单元 | 决策与职责 | 拥有的状态 | 一个必须成立的不变量 |
| --- | --- | --- | --- |
| | | | |
| | | | |
| | | | |

画出它们之间的主要关系。可使用文字、ASCII 或 Mermaid。

<!-- 作者绘制，不预填参与者与边界 -->

## B. 画正常请求时序：12 分钟

继续凭理解画出请求从进入到结束的过程：

- 自己选择参与者，在箭头上写出传递的数据或命令。
- 标出至少两次状态变化。
- 标出一处资源归属与释放或继续保留的时机。
- 用问号标出一次目前解释不清的交互。

<!-- 作者绘制，不预填执行步骤 -->

用 100–200 字说明：为什么这样划分职责，以及图里哪个边界最难判断。

<!-- 作者填写 -->

### 作者口述初稿（原文，尚未修订）

以下保留聊天原文。原练习关闭前缀复用，作者额外讨论了启用前缀复用时的路径，审阅时分别标明条件。

> 也就是说，全书采用自顶而下加索引跳转的架构，以叙事的形式展开对吧。首先也是最重要的就是一条请求从收到到返回，经历了什么。我凭着记忆回答，有不对的地方告诉我，这个可以成为引言。首先是通过rust的server接收到一条http请求，采用axum框架，然后是路由给专门的处理函数，应该是异步的+同步的架构，还是什么忘了，然后发给scheduler进行调度，scheduler首先是经过队列管理，会按照当前kvcache容量或worker节点的状态判断是否要把这个等待队列的请求转移到运行队列并发给对应的worker，并附带有前缀复用的信息。worker收到后，就通过kvcache分配器，给这一批请求分配单个位置，每个请求一个位置，构建元数据，然后丢给modelrunner。这里是异步的，modelrunner检测到有数据进来后，就执行，严格按照元数据的法则执行cudagraph即可。无脑执行，是无状态执行器。与此同时，worker会继续接受新的请求并为下一批进行预调度分配资源。同时如果存在上一次执行完毕的结果，就从结果里面取出并返回给scheduler。不过好像应该是一个线程用于读取和预处理，一个线程用于返回，这样不会打架。scheduler按照请求号收到后，就返回给server，server再返回给用户。这样是流式请求。

## C. 核对两处判断：5 分钟

现在打开源码，只核对两处最不确定的判断。保留 A/B 原稿，将发现写在这里。

| 原先判断或疑问 | 查阅文件与符号 | 观察结果 | 是否修正理解 / 仍待确认 |
| --- | --- | --- | --- |
| | | | |
| | | | |

若五分钟内未找到，记录尝试过的搜索词或入口即可。

## D. 提交时的三句话

- 我最确定的判断是：
- 我最不确定的边界是：
- 我解释不清的一次交互是：

完成 A–D 后即可提交；允许尚未找到源码证据。可以直接编辑本页，也可以把草稿发在聊天中。

## E. 助手审阅区

### 第一轮反馈

核对范围：当前工作区的普通 dense 文本服务路径，重点是 TP=1 的正常执行。此次只读核对源码，未运行服务或性能实验。

可以保留的主干：Axum 接收请求，Server 与 Scheduler 通信，Scheduler 执行准入与 prefill 调度，Worker 组批、管理 KV 并执行，结果经 Scheduler 回到 Server，最终向用户输出。作者也已经指出了 GPU 工作与 CPU 准备/发送重叠的方向。

需要校正或补齐：

| 原稿中的判断 | 当前源码中的行为 | 源码入口 |
| --- | --- | --- |
| Server 是“异步的 + 同步的架构” | Axum handler 是 async；文本模板与 tokenizer 编码放在 `spawn_blocking` 中；ZMQ client 使用独立通信线程，通过 channel 连接异步 handler。传给 Scheduler 的核心输入已是 token IDs 与生成参数。 | [chat.rs](../../../crates/infer-server/src/api/openai/chat.rs)、[zmq_client.rs](../../../crates/infer-server/src/client/zmq_client.rs) |
| 等待队列转运行队列后执行 | 高层方向成立。需展开 queued/prefilling/decoding 状态，以及计算 token、序列数、KV 与 tile 等预算。中央策略安排新/续 prefill，持续 decode 由 Worker 内部驱动。启用前缀缓存时，规划阶段可携带复用索引。 | [continuous_batching.rs](../../../crates/infer-scheduler/src/domain/policy/continuous_batching.rs)、[planning.rs](../../../crates/infer-scheduler/src/application/planning.rs) |
| 每个请求分配一个 KV 位置 | 当前 allocator 的单位是 token slot。Prefill 为本次新计算的 token 分配多个 slot；普通 decode 每个活动序列本步输入一个 token，通常各新增一个 slot。请求在 batch 中的行与整条序列持有的 KV slots 是不同概念。这里讨论逻辑归属，实际 allocator 还可能有下一步预留。 | [global_kv_alloc.rs](../../../crates/infer-worker/src/domain/global_kv_alloc.rs)、[worker_scheduler.rs](../../../crates/infer-worker/src/application/worker_scheduler.rs) |
| 独立 ModelRunner 检测数据后执行 | 当前对应的实际对象是 `Runtime`。普通单卡主路径由 serve loop 直接调用；接收、控制处理、执行推进和发送由这个循环组织。异步 GPU 提交允许 CPU 在 GPU 工作期间继续处理其他事情。 | [serve_loop.rs](../../../crates/infer-worker/src/application/serve_loop.rs)、[decode_engine.rs](../../../crates/infer-worker/src/application/decode_engine.rs) |
| 按元数据执行 CUDA Graph，是无状态执行器 | Runtime 按执行计划运行，维护模型、KV pool、索引、workspace、Graph 与 ABC 等持久状态。Graph 可用性与执行形状决定 Graph/eager 路径。职责分离意味着执行层无需决定中央准入策略，不能据此推断它没有状态。 | [runtime/mod.rs](../../../crates/infer-worker/src/application/runtime/mod.rs)、[runtime/abc_decode.rs](../../../crates/infer-worker/src/application/runtime/abc_decode.rs) |
| 一个线程接收和预处理，一个线程返回，避免打架 | 当前普通 TP1 主 serve loop 没有这样拆分业务线程。纯 decode 稳态实际代码是 finalize/commit N → issue N+1 → send N；GPU N+1 可与 send N 和后续循环的 CPU 工作重叠。当前同次 commit N 在 issue N+1 之前；mixed 路径有自己的顺序。 | [DecodeEngine::run_step / finalize_pending](../../../crates/infer-worker/src/application/decode_engine.rs) |
| 返回给 Server 后就是流式输出 | 还需补上 token IDs 到文本增量的解码、停止处理、SSE 编码与终止事件。网络逐段返回是接口行为，不要求一 token 对应一段 SSE 文本。 | [streaming.rs](../../../crates/infer-server/src/api/openai/streaming.rs)、[decoder.rs](../../../crates/infer-server/src/api/openai/decoder.rs) |

阅读材料提醒：[worker_batch_design.md](../../../assets/worker_batch_design.md)已标为历史设计稿；其中 Server/Runner 两线程叙述不能直接当作当前主路径的线程图。

本轮重点是 KV 粒度和 CPU/GPU 时间关系，其余点供查阅。原稿保留在上文；后续按作者要求形成的编辑版本见[引言](../02-introduction.md)。

### 给作者的两道追问

1. 设 prompt 含 3 个 token，关闭前缀复用。Prefill 刚产生第一个输出 token 时，逻辑上已写入 KV 的是哪些 token？下一次普通 decode 消费哪个 token，并新增谁的 KV？先忽略下一步预分配。
2. 画两条时间线（Worker 主线程、GPU）：标出上一轮结果完成、下一轮提交、上一轮结果发送。说明为什么一个主线程也能做到 CPU/GPU 重叠，并指出何时仍必须等待 GPU。

## F. 作者修订区

<!-- 审阅后由作者修订，并记录与原先理解的差别。 -->

## G. 第一节的范围

编辑进展（2026-09-21）：根据作者明确要求，将口述主线经事实修正和文字润色写入[引言：一条推理请求的旅程](../02-introduction.md)，补充请求流程图与源码索引。该正文由作者口述、助手编辑形成；作者独立修订区继续保留。

目录进展（2026-09-24）：[全书目录](../01-CONTENTS.md)将下一章确定为“请求与会话的生命周期”，先从[1.1 节：请求、会话与序列的身份](01-request-identity.md)展开。资源布局与回收在 Runtime 篇继续深入。

返回[共写入口](../00-README.md)。
