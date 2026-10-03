# 深入理解 RustInfer

本书从一条推理请求出发，沿 Server、Scheduler、Worker 三个进程进入 Worker 内部的模型执行器（Model Runner，源码类型为 `Runtime`），再深入 CUDA 算子与张量并行。读者可以沿请求链理解系统，也可以按目录进入资源管理、GPU 执行和多 rank 协作等主题。

## 阅读入口

- [全书目录](01-CONTENTS.md)：六篇内容的章节安排、知识归属与阅读路线。
- [引言：一条推理请求的旅程](02-introduction.md)：建立 Server、Scheduler、Worker 与 GPU 的整体关系。
- [第 1 章：请求与会话的生命周期](chapters/requests/01-lifecycle.md)：从身份、状态和位置建立全局认识，解释职责边界、正常结束与取消。
- [第 2 章：HTTP、异步任务与 ZMQ 通信](chapters/requests/02-server-and-transport.md)：输入编码、轻量唤醒、结果接收与 SSE、各层请求表，以及线程通信和锁的边界。
  - [结果接收与 SSE](chapters/requests/02-server-and-transport.md#receiving-results)：从 Worker 批次结果到每请求文本输出。
  - [请求状态与结果关联](chapters/requests/02-server-and-transport.md#request-state-and-routing)：pending、会话表、序列状态，以及与 vLLM 的结构对照。
- [第 3 章：Scheduler——面向 Worker Group 的通信与资源调度](chapters/requests/03-scheduler-and-worker-group.md)：从模型实例的执行单位出发，解释中央调度、Worker 本地推进与资源归属。
  - [连接、容量报告与就绪](chapters/requests/03-scheduler-and-worker-group.md#connection-and-readiness)：三条消息链路、Hello 到 Ready、控制调用与运行期心跳。
  - [请求接纳与 Engine 事件循环](chapters/requests/03-scheduler-and-worker-group.md#request-admission)：输入校验、会话建档、请求表与事件驱动调度。
- [第 4 章：Worker 全貌与服务循环](chapters/worker/04-worker-service-loop.md)：沿 A 正在 Decode、B 的 Prefill 到达，串起职责、Group 协作、通信、状态、执行计划与结果回传。
  - [Worker Group 内部协作](chapters/worker/04-worker-service-loop.md#worker-and-group)：服务层、Model Runner、core、模型、后端与 TP rank 的关系。
  - [控制面与数据面](chapters/worker/04-worker-service-loop.md#control-and-data)：命令、结果与控制消息如何进入服务循环。
  - [从命令到执行计划](chapters/worker/04-worker-service-loop.md#command-to-plan)：Worker 组装 StepRequest，Runtime 生成 BatchPlan 并选择执行路径。
- [第 5 章：从批次命令到执行计划](chapters/worker/05-command-to-plan.md)：从分段输入与前缀提示，推导混合批次、设备索引与执行契约。
  - [逐 token 的位置映射](chapters/worker/05-command-to-plan.md#batch-plan-and-index)：区分序列行、token 行、逻辑位置与 KV slot，解释长度、分块与填充。
  - [执行与工作区契约](chapters/worker/05-command-to-plan.md#execution-contract)：形状校验、执行路径、工作区与模型上下文怎样配合。
- [第 6 章：KV Cache 的物理布局与所有权](chapters/worker/06-kv-layout-and-ownership.md)：推导 KV 容量与物理地址，沿 slot 的分配、预留、提交和回收理解资源生命周期。
  - [各层张量与物理寻址](chapters/worker/06-kv-layout-and-ownership.md#physical-layout)：slot 怎样连接逻辑位置与每层 K/V，行重排为什么无需搬动缓存。
  - [分配器怎样联动显存](chapters/worker/06-kv-layout-and-ownership.md#allocator-to-device)：沿实际结构和代码，从借出编号进入设备索引与 kernel 写入。
  - [slot 的使用权与归还](chapters/worker/06-kv-layout-and-ownership.md#slot-ownership)：区分张量存储、分配器账本、序列状态、临时 lease 与共享前缀。
  - [简单分配器与懒回收](chapters/worker/06-kv-layout-and-ownership.md#allocator-structure)：展开四个字段和分配回收代码，用八 slot 账本推演游标与待回收列表。
  - [容量压力与在途访问](chapters/worker/06-kv-layout-and-ownership.md#capacity-and-inflight)：回收如何配合控制面，以及何时才能安全复用设备存储。
- [第 10 章：CUDA Graph 与动态批次](chapters/worker/10-cuda-graph-and-dynamic-batching.md)：用实际结构和 kernel 解释固定地址如何承接变化的请求，展开捕获、分桶、补齐、资源存活与回退。
  - [五条请求使用八行 Graph](chapters/worker/10-cuda-graph-and-dynamic-batching.md#five-to-eight)：逐项追踪长度、KV 索引、写入判断和真实输出边界。
  - [几种不同的 padding](chapters/worker/10-cuda-graph-and-dynamic-batching.md#padding-kinds)：区分零长度尾部、临时 KV lease 和 token 补齐。
  - [Decode、Prefill 与 mixed](chapters/worker/10-cuda-graph-and-dynamic-batching.md#execution-paths)：解释图内范围、多维 bucket 与模型、TP 的适用条件。
  - [预热与资源生命周期](chapters/worker/10-cuda-graph-and-dynamic-batching.md#lifecycle-and-fallback)：连接 arena、地址存活、捕获失败与正常 eager 回退。
- [第 13 章：GPU 执行模型与成本分析](chapters/cuda/13-gpu-execution-and-cost.md)：从线程分工、存储层级与延迟隐藏，进入成本推导、时间线和算子指标。
  - [GPU 执行模型](chapters/cuda/13-gpu-execution-and-cost.md#gpu-execution-model)：grid、block、warp 与 SM 怎样组织一次计算。
  - [成本模型](chapters/cuda/13-gpu-execution-and-cost.md#cost-model)：结合形状推导计算量、访存量与计算强度。
  - [Nsight Systems 时间线](chapters/cuda/13-gpu-execution-and-cost.md#nsight-systems)：区分主机调用、设备执行、排队、同步与重叠。
  - [Nsight Compute 指标](chapters/cuda/13-gpu-execution-and-cost.md#nsight-compute)：解释 occupancy、吞吐、warp stalls 与 roofline 的含义和判断边界。
  - [概念索引](chapters/cuda/13-gpu-execution-and-cost.md#concept-index)与[源码索引](chapters/cuda/13-gpu-execution-and-cost.md#source-index)：从术语回到解释与项目入口。
- [第 17 章：从矩阵乘法推导 TP](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)：从两 rank 矩阵手算进入层内切分，连接权重布局、KV、词表与通信成本。
  - [列并行与行并行](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#column-and-row)：区分完整特征片段与部分和，说明拼接、归约和 bias 的位置。
  - [Attention、FFN 与词表](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#transformer-and-vocab)：沿一层计算解释本地 head、KV、gate/up 和 logits。
  - [checkpoint 与量化边界](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#checkpoint-and-quantization)：QKV 先切后拼、模型本地维度与 FP8 块对齐。
  - [显存与通信量](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#compute-memory-communication)：推导每 rank 的计算、缓存与传输，解释 TP 扩展收益的条件。
- [项目运行说明](../../README.md)：构建、模型配置与服务启动。
- [专题 A：推测解码](topics/01-speculative-decoding.md)：从 pending 与 K+1 行进入一轮完整执行，解释各组件职责、提案、验证、状态恢复与返回。
  - [组件与状态归属](topics/01-speculative-decoding.md#components)：服务适配器、proposer、head、Runtime、verifier 与 committer 如何协作。
  - [MTP、EAGLE3 与 DFlash](topics/01-speculative-decoding.md#draft-generation)：特征对齐、逐 token 提案、整块提案与设备 token tape。
  - [恢复、追赶与提交](topics/01-speculative-decoding.md#state-and-commit)：以部分接受的账本串起 target KV、recurrent state、draft KV 和 lease。
  - [结果回到用户](topics/01-speculative-decoding.md#output-and-termination)：多 token 结果、停止条件、SSE 与取消边界。

先读引言，再沿 Server → Scheduler → Worker 逐步展开。进入 Worker 时先建立服务循环与状态归属的全貌，再按需深入 Tensor、模型计算和 Rust 所有权；阅读 CUDA 与 TP 时，再补齐相应的数学与执行模型。

## 共写约定

正文面向读者，直接叙述系统行为、机制、因果关系和适用条件。修改建议、实现者提醒、聊天中的纠错过程与作者原话放在 `workshops/`，不写进正文；正文不附核对日期、版本快照或学习进度评价。

作者通过自己的解释参与写作：先口述一段执行过程或画一张图，再围绕不清楚的地方查源码、讨论和推导。助手帮助定位实现、解释原理、组织内容与润色；讨论之后，作者可以用自己的话重新讲一遍，或推演一个改变条件的场景。每次围绕一个小问题展开，不规定完成篇幅和时限，也不安排实验作业。

共写材料：

- [领域小节模板](templates/00-domain-section.md)：组织解释、图示、推导和源码依据。
- [专题 A 共写记录](workshops/01-topic-a-speculative-decoding.md)：亲自复述一轮推测，推导拒绝与 EOS 的状态账本，解释两类 proposer 的对齐、恢复和收益条件。
- [第 17 章共写记录](workshops/17-matmul-to-tensor-parallel.md)：亲自拆分矩阵、追踪一层 shape 与通信，再推导 KV、量化块和每 rank 的成本。
- [第 13 章共写记录](workshops/13-gpu-execution-and-cost.md)：亲自解释线程分工、occupancy 与执行时间的关系，推导矩阵形状变化和 kernel 重叠。
- [第 10 章共写记录](workshops/10-cuda-graph-and-dynamic-batching.md)：保留五行用八行图的提问，亲自解释有效长度、地址更新、旧尾部失效与 mixed 分桶。
- [第 6 章共写记录](workshops/06-kv-layout-and-ownership.md)：亲自推导容量和分配器字段变化，解释懒回收、显存联动、预留与共享的归还条件。
- [第 5 章共写记录](workshops/05-command-to-plan.md)：亲自推导分段输入、位置映射和行重排，再解释各种计划的职责。
- [第 4 章共写记录](workshops/04-worker-service-loop.md)：围绕控制面、数据面、Group 协作与 A、B 的执行过程继续解释。
- [第 3 章共写记录](workshops/03-scheduler-and-worker-group.md)：保留多机器资源调度的设计动机，继续展开职责解释与预算推演。
- [第 2 章共写记录](workshops/02-server-and-transport.md)：保留作者原话，以及关于异步提交和线程通信的复述、推导话题。
- [请求身份与位置](workshops/01-request-identity.md)：区分请求、会话、序列、batch row 与 KV slot。
- [最初的领域边界与请求时序](workshops/00-domain-map.md)：保留最初的口述与讨论。

## 用职责与不变量组织内容

领域小节围绕五个问题展开：

1. 它解决什么问题，负责哪些决策？
2. 它拥有什么状态，谁有权修改？
3. 哪些不变量必须成立，哪些变化需要一起提交？
4. 它通过什么接口、命令或事件与其他部分协作？
5. 正常结束、取消、失败和迟到结果分别如何处理？

领域边界根据职责、状态归属与一致性要求判断，再映射到 crate、模块和符号。Domain、Application、Infrastructure 表示实现分层，不能直接当作领域或限界上下文。

Tensor、CUDA 算子与 TP 同时涉及数学和执行机制，可按输入输出契约、数据布局、执行依赖与成本组织小节。跨领域请求时序把这些局部机制连接起来。

## 技术说明的依据

源码入口帮助读者理解机制和找到实现。涉及正确性与性能时，区分原理推导、已有测试和实际测量；引用数值时说明形状、硬件、精度和负载等条件，不把推断写成测量结论。验证方法、Profiling 与性能分析本身也是书中的技术内容，可以结合现有实现和资料解释。

阅读主线从第 1 章的全局对象与生命周期，进入第 2 章的 Server、第 3 章的 Scheduler，再进入[第 4 章的 Worker 服务循环](chapters/worker/04-worker-service-loop.md)、[第 5 章的批次计划](chapters/worker/05-command-to-plan.md)与[第 6 章的 KV 布局及所有权](chapters/worker/06-kv-layout-and-ownership.md)。第二篇后续安排见[Worker 篇目录](01-CONTENTS.md#worker-execution)。作者可以在聊天中解释 A、B 的资源变化，也可以在[第六章共写记录](workshops/06-kv-layout-and-ownership.md)中写下容量推导和分配回收过程，再继续深入 Tensor 与异步执行。

需要理解固定执行形状如何适应动态请求时，可以进入[第 10 章](chapters/worker/10-cuda-graph-and-dynamic-batching.md)；需要补充 GPU 原理时，可以进入[第 13 章](chapters/cuda/13-gpu-execution-and-cost.md)；需要把模型计算扩展到多个 rank 时，可以进入[第 17 章](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)，再回到对应的 Worker 机制。第 7–9、11–12 章仍按原目录逐步展开；章节编号与全书阅读顺序保持一致。
