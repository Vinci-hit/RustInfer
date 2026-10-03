# 全书目录与写作地图

本书围绕推理框架、CUDA 与张量并行展开。阅读主线由外到内：一条请求依次经过 Server、Scheduler 与 Worker，在 Worker 内部进入 Model Runner（源码类型为 `Runtime`），再深入 GPU 算子与多个 TP rank 的协作。每个具体机制再从最小原理、状态与不变量推到实现和设计取舍。

引言、第 1、2、4、5、6、10、13、17 章与专题 A 已有正文，第 3 章已写至 3.3，第 7–9、11–12 章与其余内容逐步展开。章节编号表示阅读顺序。

## 目录的四个层次

| 层次 | 组织依据 | 作用 |
| --- | --- | --- |
| 篇 | 读者要建立的一组能力 | 从请求生命周期进入 Worker、CUDA、TP，再到验证与工程实践 |
| 章 | 一个能独立讨论的问题或职责范围 | 说明问题、实现、取舍和证据 |
| 节 | 一项机制、状态关系或不变量 | 围绕一个具体问题展开解释与推导 |
| 索引 | 领域职责、概念、源码符号、测试与性能资料、面试问题 | 支持跳转，并指回该知识点的正文归属 |

DDD 用来确定职责与状态边界。每个领域小节回答“谁作决策、谁拥有状态、谁保证不变量、如何协作”，再映射到 Domain / Application / Infrastructure 的实现分层。一个领域问题可能跨越多个文件或 crate；限界上下文与聚合边界需要结合一致性要求论证。

CUDA 算子、Tensor 和 TP 同时涉及数学与执行机制，其小节按输入输出契约、数据布局、执行依赖和成本展开。请求时序将这些局部分析连接起来。

一个知识点指定一处主要正文，其他章节用简短回顾与索引引用。例如，第 1 章负责请求身份，第 3 章负责 Scheduler 与资源预算，第 4 章负责 Worker 服务循环，第 6 章负责 KV slot，第 9 章负责异步完成，第 10 章负责 Graph 捕获条件。引言负责串起全程，正文逐章放大局部。

## 阅读起点

- [引言：一条推理请求的旅程](02-introduction.md)：建立整体执行链。
- [第 1 章：请求与会话的生命周期](chapters/requests/01-lifecycle.md)：区分身份、状态与位置，建立三个进程及 Worker 内部服务层与模型执行器的职责关系。
- [第 2 章：HTTP、异步任务与 ZMQ 通信](chapters/requests/02-server-and-transport.md)：从 Server 输入处理进入 channel、线程协作与跨进程消息，再沿结果接收、请求分发回到 SSE。
- [第 3 章：Scheduler——面向 Worker Group 的通信与资源调度](chapters/requests/03-scheduler-and-worker-group.md)：建立 Group、中央调度与 Worker 本地执行的职责关系，沿请求展开资源协调。
- [第 4 章：Worker 全貌与服务循环](chapters/worker/04-worker-service-loop.md)：将职责、通信、组内协作、状态与执行计划连接成完整服务循环。
- [第 5 章：从批次命令到执行计划](chapters/worker/05-command-to-plan.md)：从片段接续与混合组批，推导逐 token 的位置映射、设备索引和执行契约。
- [第 6 章：KV Cache 的物理布局与所有权](chapters/worker/06-kv-layout-and-ownership.md)：从每 token 字节数进入各层张量，追踪 slot 的借出、提交、共享与安全回收。
- [第 10 章：CUDA Graph 与动态批次](chapters/worker/10-cuda-graph-and-dynamic-batching.md)：从五条请求使用八行 Graph，解释固定地址、有效长度、KV 隔离、mixed 分桶、捕获与回退。
- [第 13 章：GPU 执行模型与成本分析](chapters/cuda/13-gpu-execution-and-cost.md)：连接线程分工、内存层级、延迟隐藏、成本模型与 Nsight 观测，为 CUDA 算子分析建立基础。
- [第 17 章：从矩阵乘法推导 TP](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)：从拼接与部分和进入模型切分，推导权重加载、KV、量化边界与通信成本。
- [专题 A：推测解码](topics/01-speculative-decoding.md)：沿一轮提案、验证、恢复、追赶与提交，解释 MTP、EAGLE3、DFlash 的组件职责和两份 KV 的变化。
- 模型、硬件、构建和运行步骤参考[项目 README](../../README.md)，验证范围参考[验证文档](../VALIDATION.md)。
- 前置知识按需补充：Rust 所有权与并发、矩阵与 Transformer、CUDA 基础分别在相关章节给出最小铺垫，再由概念索引串联。

## 第一篇：请求的身份、状态与协作边界

本篇把引言中的参与者落实为具体对象、状态和接口，用请求状态图与职责图连接后续章节。

| 章 | 小节安排 | 讲解重点与图示 |
| --- | --- | --- |
| **[1. 请求与会话的生命周期](chapters/requests/01-lifecycle.md)** | [1.1 请求的身份](chapters/requests/01-lifecycle.md#identity)；[1.2 从等待到生成](chapters/requests/01-lifecycle.md#lifecycle)；[1.3 状态归属与 DDD](chapters/requests/01-lifecycle.md#ownership)；[1.4 结束、取消与迟到结果](chapters/requests/01-lifecycle.md#termination) | 对象与身份表、批次行变化示例、状态迁移图、取消时间线 |
| **[2. 从接口到领域：HTTP、异步任务与 ZMQ 通信](chapters/requests/02-server-and-transport.md)** | 2.1 HTTP 到 token IDs；2.2 async 与线程；2.3 channel 与流式提交；2.4 ZMQ 分层与协议；2.5 轻量唤醒与同步收发；[2.6 结果接收与 SSE](chapters/requests/02-server-and-transport.md#receiving-results)；[2.7 请求状态与结果关联](chapters/requests/02-server-and-transport.md#request-state-and-routing)；2.8 收益和边界；2.9 计网、OS、锁与环形队列；2.10 DDD 与源码 | 请求与结果双向链路图、每请求 channel、身份与状态表、与 vLLM 对照、等待与内存序、源码分层映射 |
| **[3. Scheduler：面向 Worker Group 的通信与资源调度](chapters/requests/03-scheduler-and-worker-group.md)** | [3.1 Scheduler 与 Worker Group 的职责](chapters/requests/03-scheduler-and-worker-group.md#responsibilities)；[3.2 连接、容量报告与就绪](chapters/requests/03-scheduler-and-worker-group.md#connection-and-readiness)；[3.3 请求接纳与 Engine 事件循环](chapters/requests/03-scheduler-and-worker-group.md#request-admission)；3.4 token、序列、KV 与在途预算；3.5 连续批处理、chunked prefill 与公平性；3.6 计划、派发与 Worker 本地执行；3.7 结果反馈与下一轮调度；3.8 取消、资源压力与故障；3.9 多机器与多 Group 扩展 | Group 与职责图、启动和双向通信时序、预算账本、选批推演、结果反馈闭环；会话、资源、策略的 DDD 分层与源码索引 |

第 1 章建立全局对象与生命周期，第 2 章进入 Server，第 3 章沿请求进入 Scheduler，第 4 章承接派发后的 Worker 服务循环。KV 的物理布局在第 6 章展开，前缀共享与所有权进入第 12 章，CPU/GPU 的执行重叠进入第 9 章。

<a id="worker-execution"></a>

## 第二篇：Worker——从批次命令到 GPU 执行

本篇沿一个批次进入 Worker：先假设模型已经加载、Worker 已经就绪，从接收命令与服务循环开始，依次展开执行计划、资源所有权、计算接口、模型计算与异步执行。理解运行过程后，再回看这些资源如何在启动阶段建立，最后串起跨进程的前缀复用。

从[第 4 章](chapters/worker/04-worker-service-loop.md)进入 Worker Server、Model Runner、core、模型组件与后端的职责，再沿控制面、数据面和本地状态走到实际执行。组内 TP 协作在开篇建立关系，在第四篇深入展开。

| 章 | 小节安排 | 讲解重点与图示 |
| --- | --- | --- |
| **[4. Worker 全貌与服务循环](chapters/worker/04-worker-service-loop.md)** | [4.1 职责与 Group 协作](chapters/worker/04-worker-service-loop.md#worker-and-group)；[4.2 控制面与数据面](chapters/worker/04-worker-service-loop.md#control-and-data)；[4.3 序列、行序与在途步骤](chapters/worker/04-worker-service-loop.md#worker-state)；[4.4 从批次命令到执行计划](chapters/worker/04-worker-service-loop.md#command-to-plan)；[4.5 完成、状态更新与回传](chapters/worker/04-worker-service-loop.md#completion-and-output)；[4.6 空闲、取消与结束](chapters/worker/04-worker-service-loop.md#idle-and-termination) | A、B 场景、职责与 Group 图、消息分类、状态表、小批次元数据及执行时序；串起服务层决策、Runtime 计划与协议结果 |
| **[5. 从批次命令到执行计划](chapters/worker/05-command-to-plan.md)** | [5.1 命令、前缀信息与 Prefill 分段](chapters/worker/05-command-to-plan.md#command-and-segments)；[5.2 SeqStep、StepRequest 与混合批次](chapters/worker/05-command-to-plan.md#step-request-and-mixed-batch)；[5.3 逻辑序列、物理行序与 BatchPlan 元数据](chapters/worker/05-command-to-plan.md#batch-plan-and-index)；[5.4 ExecutionPlan、形状校验与 workspace 契约](chapters/worker/05-command-to-plan.md#execution-contract) | A、B 完整推导、分段与前缀示例、逐 token 索引映射、Q tile、行重排与三种填充；区分实际计划、运行形状与资源契约 |
| **[6. KV Cache 的物理布局与所有权](chapters/worker/06-kv-layout-and-ownership.md)** | [6.1 KV 容量推导](chapters/worker/06-kv-layout-and-ownership.md#kv-capacity)；[6.2 token slot、block table 与 batch row](chapters/worker/06-kv-layout-and-ownership.md#physical-layout)；[6.3 分配、预留、提交与回收](chapters/worker/06-kv-layout-and-ownership.md#slot-ownership)；[6.4 容量不足与设备在途访问](chapters/worker/06-kv-layout-and-ownership.md#capacity-and-inflight) | 实际数据结构与代码、四字段分配器、八 slot 懒回收账本；从 CPU 编号经设备索引进入 K/V 写入，区分 slot、Storage 与 CUDA allocation 的回收 |
| **7. infer-core：Tensor、存储与后端契约** | 7.1 core 与 Worker、backend 的依赖关系；7.2 shape、stride、offset 与 view；7.3 dtype、设备与存储所有权；7.4 执行接口、算子 trait 与通用计划；7.5 泛型分派与 unsafe / FFI 契约 | 共享存储与张量视图图、接口到后端的调用链、一个别名或生命周期案例 |
| **8. 一步模型计算** | 8.1 Prefill 与 Decode 的输入输出；8.2 embedding、attention、FFN 与残差；8.3 logits、采样与下一步输入；8.4 模型组件与算子调用链 | 手推一个小模型的 shape，解释“输入 token 的 KV”与“输出 token”的时间关系 |
| **9. 异步执行与 ABC 流水线** | 9.1 issue、finalize 与逻辑提交；9.2 H2D、计算、D2H 的依赖；9.3 ABC 缓冲、持久地址与行压缩；9.4 上一步结果回传与下一步提交；9.5 新请求、取消和失败时的收尾 | CPU/GPU 时间线，区分提交、完成、更新状态与发送结果；推演缓冲复用和过早释放的风险 |
| **[10. CUDA Graph 与动态批次](chapters/worker/10-cuda-graph-and-dynamic-batching.md)** | [10.1 捕获结构与持久地址](chapters/worker/10-cuda-graph-and-dynamic-batching.md#capture-and-addresses)；[10.2 bucket、padding 与有效行](chapters/worker/10-cuda-graph-and-dynamic-batching.md#buckets-and-padding)；[10.3 Decode、Prefill、mixed 的执行条件](chapters/worker/10-cuda-graph-and-dynamic-batching.md#execution-paths)；[10.4 预热、资源生命周期与 eager 路径](chapters/worker/10-cuda-graph-and-dynamic-batching.md#lifecycle-and-fallback) | 五行用八行 Graph 的完整索引与 kernel 推演；区分零长度尾部与临时 KV Pad，展开 mixed 多维分桶、捕获范围、arena 与回退边界 |
| **11. 从权重文件到可服务的 Worker** | 11.1 配置、模型组装与后端选择；11.2 权重读取、转换与上传；11.3 KV pool、工作区与显存规划；11.4 Graph 预热、自检与 Ready | 回看前面各章资源的建立过程；加载阶段图与内存账本 |
| **12. 前缀复用与跨进程资源协作** | 12.1 Scheduler 的 RadixTree 匹配与节点切分；12.2 前缀提示、执行结果与物理 KV 索引；12.3 owner、pin 与共享索引；12.4 缓存保留、淘汰与安全释放；12.5 命中边界与收益条件 | 两条共享前缀请求的所有权图，串起匹配、复用、完成和回收的跨进程消息 |

<a id="worker-service-loop"></a>

### 从第 4 章进入 Worker

[第 4 章正文](chapters/worker/04-worker-service-loop.md)以“A 正在 Decode，B 的 Prefill 到达”为主线，整合 Worker 的职责分层、控制面与数据面、Group 内部协作、状态管理、命令到计划的转换，以及完成与回传。先跟随一个四 token 的 Prefill 与一次 Decode 组成逻辑批次，再解释执行重叠、空闲与取消。

[第 5 章](chapters/worker/05-command-to-plan.md)深入批次元数据与计划契约，[第 6 章](chapters/worker/06-kv-layout-and-ownership.md)深入 KV 的物理布局、容量与所有权，第 9 章展开异步依赖，[第 10 章](chapters/worker/10-cuda-graph-and-dynamic-batching.md)解释动态批次如何安全重放 Graph。第 3 章负责 Scheduler 的预算与派发，第 4–5 章负责 Worker 根据本地状态组织执行，两侧通过命令、完成结果与容量信息协作。

第 3 章解释连接、容量与就绪；第 11 章回看模型加载、资源初始化和计算自检；第 22 章展开外部观测与故障诊断。A、B 的作者复述与后续讨论保存在[第四章共写记录](workshops/04-worker-service-loop.md)。

<a id="cuda-execution"></a>

## 第三篇：CUDA 算子与性能成本

这一篇从 Worker 的模型计算与后端调用继续向下，解释数学如何映射到线程、内存和 GPU 指令。性能结论同时给出形状、硬件、精度与测量条件。

| 章 | 小节安排 | 讲解重点与图示 |
| --- | --- | --- |
| **[13. GPU 执行模型与成本分析](chapters/cuda/13-gpu-execution-and-cost.md)** | [13.1 GPU 执行模型](chapters/cuda/13-gpu-execution-and-cost.md#gpu-execution-model)；[13.2 内存层级与访问](chapters/cuda/13-gpu-execution-and-cost.md#memory-and-access)；[13.3 occupancy 与延迟隐藏](chapters/cuda/13-gpu-execution-and-cost.md#occupancy-and-latency)；[13.4 成本模型](chapters/cuda/13-gpu-execution-and-cost.md#cost-model)；[13.5 launch、stream 与 Graph](chapters/cuda/13-gpu-execution-and-cost.md#launch-stream-and-graph)；[13.6 Nsight Systems 时间线](chapters/cuda/13-gpu-execution-and-cost.md#nsight-systems)；[13.7 Nsight Compute 指标](chapters/cuda/13-gpu-execution-and-cost.md#nsight-compute)；[13.8 从瓶颈到优化](chapters/cuda/13-gpu-execution-and-cost.md#bottlenecks-and-optimization) | 线程分工、存储与执行依赖图；计算量和访存量推导、时间线与指标解读；用证据连接瓶颈判断和优化方向 |
| **14. GEMM、逐元素算子与融合** | 14.1 推理中的矩阵形状；14.2 tiling、Tensor Core 与布局；14.3 RMSNorm、RoPE、激活及融合；14.4 库实现、自定义 kernel 与分派取舍 | 一条算子调用链、一个代表性 kernel 的线程与数据布局图 |
| **15. Paged Attention 的 Prefill 与 Decode** | 15.1 计算与缓存读取差异；15.2 分块计算与 softmax 归约；15.3 paged KV 索引、ragged batch 与 mask；15.4 边界形状、数值误差与性能 | 手推小 attention、索引示意图、典型与边界形状的执行分析 |
| **16. 量化格式与执行路径** | 16.1 权重格式、scale 与精度；16.2 AWQ、FP8 等路径的加载和布局；16.3 解量化、融合与算子选择；16.4 内存收益、误差和性能条件 | 一种格式的字节数推导、误差对照与执行路径说明 |

Rust 对存储所有权提供的保证在第 7 章说明；设备异步使用期间的生命周期由第 9 章展开。权重量化的执行分析集中在第 16 章，KV 量化作为独立专题讨论。

第 13 章的[概念索引](chapters/cuda/13-gpu-execution-and-cost.md#concept-index)与[源码索引](chapters/cuda/13-gpu-execution-and-cost.md#source-index)提供术语和实现入口。算子的执行成本在本篇展开，服务负载、TTFT、TPOT 与尾延迟的评测方法进入第 21 章。

<a id="tensor-parallel"></a>

## 第四篇：张量并行

主线围绕项目的单机 TP 实现，将第二篇的 Worker 执行过程扩展到多个 rank，分别解释数学、执行与故障。多节点、PP、DP、EP 的相关概念放在扩展讨论中，并注明实现范围。

| 章 | 小节安排 | 讲解重点与图示 |
| --- | --- | --- |
| **[17. 从矩阵乘法推导 TP](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)** | [17.1 列并行与行并行](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#column-and-row)；[17.2 Attention、FFN 与词表切分](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#transformer-and-vocab)；[17.3 checkpoint、QKV 与量化块边界](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#checkpoint-and-quantization)；[17.4 计算量、显存和通信量](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#compute-memory-communication) | 两 rank 数值重构与整层 shape；区分特征分片、部分和、复制张量，展开 QKV 先切后拼、GQA、词表、FP8 和通信账本 |
| **18. 多 Rank Runtime 与集合通信** | 18.1 rank、设备、线程与模型资源；18.2 NCCL communicator 与 collective；18.3 leader/follower 操作镜像；18.4 batch、KV、采样和操作顺序的一致性 | 两 rank 时间线、一轮推理的命令与 collective 对照 |
| **19. TP 的 Graph、故障与性能边界** | 19.1 每 rank 捕获与重放；19.2 issue/finalize 和跨 rank 完成；19.3 超时、错误传播与整组失败；19.4 TP1/TP2 正确性与性能比较 | 故障推演、双卡一致性条件与通信成本分析 |

[第 17 章](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)先解释一个 Group 如何共同算出同一次模型前向，再由第 18 章进入 rank 线程、命令镜像与 NCCL 的实际执行。术语与实现可从第 17 章的[概念索引](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#concept-index)和[源码索引](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md#source-index)跳转；自己的手算与解释保存在[第十七章共写记录](workshops/17-matmul-to-tensor-parallel.md)。

## 第五篇：验证、观测与工程复盘

本篇解释如何判断推理结果是否正确、如何定位瓶颈，以及如何将观测结果放回服务负载中理解。通过已有测试、指标与工程案例，连接原理、实现和结论。

| 章 | 小节安排 | 讲解重点与图示 |
| --- | --- | --- |
| **20. 正确性验证体系** | 20.1 领域不变量与状态测试；20.2 CPU 参考、GPU 算子与数值误差；20.3 模型轨迹与服务回归；20.4 分层证据、边界输入与未验证范围 | 一份测试矩阵、一个能暴露实际错误的输入 |
| **21. Profiling 与性能评测** | 21.1 TTFT、TPOT、ITL 与吞吐；21.2 host 时间、GPU 时间和等待；21.3 trace 与瓶颈定位；21.4 对照、预热、负载、尾延迟和原始记录 | 评测方法、trace 解读，以及从观测到瓶颈判断的推理 |
| **22. 服务生命周期与可观测性** | 22.1 启动、自检、health 与 ready；22.2 流式结束、取消与背压；22.3 心跳、协议与故障传播；22.4 指标语义、日志与部署限制 | 一个故障场景的全链路观察、指标起止点说明 |
| **23. 优化案例与扩展实践** | 23.1 从假设到正确性验证；23.2 一个 Runtime 或加载优化；23.3 一个 kernel 或 TP 优化；23.4 新模型、算子或策略的接入过程 | 项目中的实际案例，解释问题、取舍、局限和实现来源 |

## 第六篇：进阶专题

专题按主线所需知识跳转阅读，不作为开始使用项目的前提。推测解码从第 5、8、9 章的计划、模型计算与提交过程接入，多模态从输入处理、模型组件与状态管理接入，MoE 从第 8 章的 FFN 继续展开。各篇说明实现范围与支持条件，区分完整服务、局部实现和研究方案。

| 专题 | 小节方向 | 讲解重点与图示 |
| --- | --- | --- |
| **[A. 推测解码](topics/01-speculative-decoding.md)** | [A.1 一轮推测与 pending](topics/01-speculative-decoding.md#one-round)；[A.2 组件职责](topics/01-speculative-decoding.md#components)；[A.3 启动与 Prefill](topics/01-speculative-decoding.md#startup-and-prefill)；[A.4 MTP / EAGLE3 / DFlash 提案](topics/01-speculative-decoding.md#draft-generation)；[A.5 验证计划与停止](topics/01-speculative-decoding.md#target-verification)；[A.6 恢复、追赶与提交](topics/01-speculative-decoding.md#state-and-commit)；[A.7 返回与结束](topics/01-speculative-decoding.md#output-and-termination)；[A.8 成本与边界](topics/01-speculative-decoding.md#cost-and-limits) | K+1 行推导、部分接受与 EOS 账本、组件图与完整时序；两份 KV、shifted alignment、recurrent 恢复、device tape 和多 token 流式输出 |
| B. 混合注意力与视觉输入 | recurrent state、请求重排与复用、视觉预处理、位置和 embedding | 异构状态生命周期图 |
| C. MoE 执行 | 路由、专家计算、数据组织、单机执行与 EP 扩展的边界 | 一轮路由到输出的张量流 |
| D. DeepSeek V4 | tiny 架构参考、独立 GPU 算子、压缩与索引、模型集成边界 | 机制图和逐层验证记录 |
| E. KV 压缩与分层缓存 | KV 量化、淘汰、GPU/CPU 副本、搬运完成与收益判断 | 容量与搬运成本估算、方案契约 |

## 附录与索引

- 术语与符号：统一 request、session、sequence、row、token、slot、step、rank 等词的使用。
- 领域与源码索引：职责、主要状态、接口、正文位置和关键符号；允许多对多映射。
- 测试与性能资料索引：项目已有测试、评测入口、指标含义和结论适用范围。
- 面试索引：从已写章节提取 90 秒讲解、5 分钟展开、深挖追问和实际项目贡献。
- 实现与验证范围：区分完整服务、模型路径、独立算子、参考实现与研究方案。

## 文件如何组织

Markdown 文件名以两位序号开头，便于文件系统按名称排序。正文章文件使用全书章号；入口、目录、引言与模板使用各自目录内的排列序号，章节共写记录沿用对应章号。专题在 `topics/` 内按 `01、02…` 排列，对应正文的 A、B…；专题共写文件加入 `topic-a` 等标识，与章节记录区分。后续目录按实际成文进度创建，下面展示的是目标结构；尚未编写的章不生成空正文链接。

```text
docs/book/
  00-README.md              阅读入口与共写约定
  01-CONTENTS.md            全书目录、章节归属与写作顺序
  02-introduction.md        已有引言
  chapters/                 随成文逐步创建
    requests/               第 1–3 章，含 Server 与 Scheduler
      01-lifecycle.md
      02-server-and-transport.md
      03-scheduler-and-worker-group.md
    worker/                 第 4–12 章，服务循环、模型执行、KV 与前缀复用
      04-worker-service-loop.md
      05-command-to-plan.md
      06-kv-layout-and-ownership.md
      10-cuda-graph-and-dynamic-batching.md
    cuda/                   第 13–16 章
      13-gpu-execution-and-cost.md
    tensor-parallel/        第 17–19 章
      17-matmul-to-tensor-parallel.md
    validation/             第 20–23 章
  topics/                   进阶专题
    01-speculative-decoding.md
  indexes/                  术语、源码、测试与性能资料、面试索引
  workshops/                作者原稿、追问、修订与回忆记录
    00-domain-map.md
    01-request-identity.md
    01-topic-a-speculative-decoding.md
    02-server-and-transport.md
    03-scheduler-and-worker-group.md
    04-worker-service-loop.md
    05-command-to-plan.md
    06-kv-layout-and-ownership.md
    10-cuda-graph-and-dynamic-batching.md
    13-gpu-execution-and-cost.md
    17-matmul-to-tensor-parallel.md
  templates/                写作与练习模板
    00-domain-section.md
```

章文件采用“章号 + 语义名称”，例如 `chapters/requests/01-lifecycle.md`；第 4 章使用 `chapters/worker/04-worker-service-loop.md`。调整章号时同步修改文件名、目录与交叉链接。初期每章一个 Markdown 文件、节使用锚点；小节形成独立主题或篇幅明显影响阅读时再拆文件。测试与评测方法可引用现有 `scripts/`、`bench/` 和测试入口。

## 阅读顺序与写作顺序

**读者路线：** 快速上手先读项目运行说明、引言和第 1–4 章，串起 Server → Scheduler → Worker；深入理解沿第 5–19 章推进。已熟悉某部分的读者通过索引进入 Worker、CUDA 或 TP，并补齐对应前置知识。第 20–22 章的方法贯穿实践。

**Worker 主线：** 从第 1–3 章的身份、通信与调度进入第 4 章的服务循环，再沿第 5–10 章展开计划、KV、core、模型计算、流水线与 Graph。起步场景使用单卡、普通文本生成与关闭前缀缓存的配置；第 11 章回看启动，第 12 章引入共享前缀。先用一条持续 Decode 的 A 和一条新到达的 B 解释协作，再逐章补全具体机制。

**CUDA 与 TP：** 深入第 13–19 章，以一个代表性 CUDA 算子和一轮 TP 为例，推导形状、数据布局、执行依赖和成本，再连接源码。第 20–21 章帮助解释已有正确性与性能资料。

**专题与案例：** 衔接主线章节后，再展开进阶机制。优化案例取自项目中的实际工作，说明设计理由和适用范围。

每次围绕一个小问题，结合作者的解释、图示、源码讨论和条件推导逐步成文。通用结构参考[领域小节模板](templates/00-domain-section.md)，不必逐项套用。

## 接下来的讨论

第 1 章建立全局对象与生命周期，第 2 章串起 Server 的发送与接收。可以在[第一章共写记录](workshops/01-request-identity.md)中推演 A 结束后 B 的行号变化，也可以在[第二章共写记录](workshops/02-server-and-transport.md)中复述结果如何找到对应的响应流。

第 3 章已经展开[Scheduler 与 Worker Group 的职责](chapters/requests/03-scheduler-and-worker-group.md#responsibilities)、[连接、容量报告与就绪](chapters/requests/03-scheduler-and-worker-group.md#connection-and-readiness)，并沿请求进入[接纳与 Engine 事件循环](chapters/requests/03-scheduler-and-worker-group.md#request-admission)。作者的设计动机与推演保存在[第三章共写记录](workshops/03-scheduler-and-worker-group.md)，预算、选批、派发与结果反馈按目录保留，后续继续展开。

[第 4 章](chapters/worker/04-worker-service-loop.md)串起 Worker 的服务循环，[第 5 章](chapters/worker/05-command-to-plan.md)沿同一批次展开输入分段、位置映射、设备索引与执行契约，[第 6 章](chapters/worker/06-kv-layout-and-ownership.md)继续进入 KV 的物理布局、分配与所有权。可以在[第四章共写记录](workshops/04-worker-service-loop.md)中复述职责与执行时序，在[第五章共写记录](workshops/05-command-to-plan.md)中推导分段与行重排，再在[第六章共写记录](workshops/06-kv-layout-and-ownership.md)中解释容量账本、预留和回收。第 7 章继续深入 infer-core 的 Tensor、存储与后端契约。

[第 10 章](chapters/worker/10-cuda-graph-and-dynamic-batching.md)已经展开 Graph 捕获、动态有效长度、补齐与 KV 隔离，可以在[第十章共写记录](workshops/10-cuda-graph-and-dynamic-batching.md)中复述五行用八行图的执行过程，再推演行缩减与 mixed 形状。

[第 13 章](chapters/cuda/13-gpu-execution-and-cost.md)已经展开 GPU 执行与成本分析。可以在[第十三章共写记录](workshops/13-gpu-execution-and-cost.md)中解释 RMSNorm 的线程分工，推导矩阵行数变化的成本，再结合 occupancy 与重叠时间线说明性能判断的依据。第 7–9、11–12 章仍按既定目录逐步成文。

[第 17 章](chapters/tensor-parallel/17-matmul-to-tensor-parallel.md)已经从矩阵乘法展开模型的 TP 切分。可以在[第十七章共写记录](workshops/17-matmul-to-tensor-parallel.md)中独立完成两 rank 手算，解释一层的两个归约点，再推导词表通信与 KV 容量。第 18–19 章继续承接多 rank 的实际执行与性能边界。

返回[书籍入口](00-README.md)。
