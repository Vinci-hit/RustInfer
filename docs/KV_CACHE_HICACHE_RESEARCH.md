# KV Cache 压缩与 HiCache 调研

调研日期：2026-09-19。依据官方文档、论文、SGLang main 源码及 RustInfer 当前代码；上游 main 和在线文档会变化。本文是选型和实现建议，未实现功能、未运行性能测试。不存在覆盖所有模型、硬件和负载的“最好”算法；以下优先级按 RustInfer 的工程适配性判断，不是热度排行榜。

## 结论

第一版建议实现 **HiCache 风格的 GPU↔CPU 分层前缀缓存**：TP=1，Llama/Qwen3 普通 attention，原精度 KV，worker 内有界 pinned host pool，异步批量搬运，scheduler 管理分层索引。后续再增加 L3 存储与 FP8 KV；更低比特量化和 token 淘汰单独评估。

HiCache 扩大可复用缓存容量，减少重复 prefill，本身不减少每个 token 的 KV 字节数。完整 attention 的活跃上下文仍需能够驻留 GPU；它不是让任意长单请求在 CPU KV 上直接 decode 的方案。[SGLang 设计](https://docs.sglang.io/docs/advanced_features/hicache_design)

## 方案分类与优先级

| 方向 | 代表 | 核心收益 | RustInfer 建议 |
| --- | --- | --- | --- |
| 分层缓存、前缀复用 | HiCache、LMCache、vLLM KV offloading | 保留更多历史前缀，降低重复 prefill 和 TTFT | 第一版 |
| KV 量化 | FP8 | 相对 BF16，KV 数据主体约减半；增加驻留 token 容量 | 首选的后续压缩路线 |
| 低比特 KV 量化 | KIVI、TurboQuant | 进一步减少字节数，需专用编码和 attention 读取路径 | 后续实验 |
| token 淘汰 | H2O、SnapKV、PyramidKV、KVzip | 丢弃部分历史 KV，减少存储和 attention 访问 | opt-in 实验，先做质量评测 |
| 稀疏读取 / offload | Quest、HiSparse 等 | 少读部分 KV 或按需搬运 | 不等价于删除或量化全部 KV |
| 模型原生压缩 | MLA、模型自带压缩 attention | 从模型结构上减少状态 | 按模型实现，不能泛化为通用插件 |

FP8 已有成熟引擎文档和校准路径，但收益取决于 attention backend：不能只把缓存转成 FP8 后每步解压整段 BF16。需要明确 scale 粒度、写入量化、读取计算和模型质量；权重量化也不等于 KV 量化。[vLLM FP8 KV](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

低比特方案中，KIVI 利用 K/V 分布差异采用非对称量化；TurboQuant 是 2026 年值得关注的路线，使用旋转与量化并处理内积估计误差。论文或发布文章中的压缩比、局部 kernel 加速与质量结论，不应直接视为 RustInfer 的端到端结果。实际占用还包括 scale、残差、高精度尾部、对齐及 workspace。[KIVI](https://arxiv.org/abs/2402.02750)、[Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

SnapKV 根据 observation window 选择 KV；PyramidKV 调整不同层的预算；KVzip 针对未来不同 query 的复用设计压缩。对于多轮对话，同一前缀未来可能被问到不同内容，必须测试被淘汰信息是否会再次需要。[SnapKV](https://arxiv.org/abs/2404.14469)、[PyramidKV](https://arxiv.org/abs/2406.02069)、[KVzip](https://arxiv.org/abs/2505.23416)

token 淘汰还有系统成本：获取 attention scores 可能需要修改 FlashAttention；按 head/layer 选择 token 会增加索引复杂度；一般多-token page allocator 需要 compact 才能真正回收页。RustInfer 的 token-slot allocator 不能直接套用最后一项结论，但仍需处理实际物理回收、共享前缀与位置语义。[NVIDIA 工程分析](https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/)

LMCache 是应当对照调研的独立缓存层，提供存储和传输扩展，也有独立进程架构。若目标是快速接入外部缓存生态，应评估 connector；若目标是原生 Rust、保持当前所有权划分，则实现 HiCache 的核心机制更贴合现有结构。不能假定 Python 集成可直接用于 Rust。[LMCache](https://docs.lmcache.ai/)、[vLLM 原生 offloading](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)

## HiCache 如何工作

```text
请求 tokens → 分层前缀索引
               ├─ GPU 命中 → 直接复用
               ├─ CPU 命中 → 分配 GPU slots → H2D → 完成后发布命中
               └─ 未命中   → prefill → 将可复用 KV 异步备份至 CPU

GPU 压力 → 淘汰无请求/搬运引用的 GPU 副本，保留 CPU 副本及索引
CPU 压力 → 淘汰可回收的 CPU 副本；可选写入 L3
L3 命中  → 预取至 CPU → 回载 GPU（后续版本）
```

SGLang 的 L1 GPU、L2 host 属于推理实例；L3 是否跨实例共享取决于 backend 和部署。HiRadixTree 保留本地副本位置，L3 存在性按需查询。相同文本的复用还要求相同计算语义：模型权重、adapter、位置与输入必须匹配。[官方设计](https://docs.sglang.io/docs/advanced_features/hicache_design)

官方支持三种备份策略：`write_through` 提前备份，`write_through_selective` 按热度选择，`write_back` 在淘汰时备份。L3 预取有 `best_effort`、`wait_complete`、`timeout`；host 布局有 layer-first 和 page-first 变体，须配合传输 backend。[最佳实践](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)

源码阅读入口：

- [hiradix_cache.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py)：`write_backup`、`load_back`、回执处理、GPU/host 淘汰和节点 split。当前 write-through 路径要求备份沿根形成连续前缀；搬运中节点被 split 后也要正确发布完成状态。
- [cache_controller.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py)：批量 transfer 提交、host 分配、完成事件和 ack 队列。

核心不是增加一个 CPU 字典，而是把“副本存在”“搬运完成”“可以使用”“可以释放”分开表示。

## RustInfer 的现状与接入点

| 当前位置 | 核实到的行为 | 建议改动 |
| --- | --- | --- |
| `crates/infer-core/src/radix_tree.rs` | token 粒度索引，`global_indices` 指向 worker slots；owners pin，LRU 淘汰 | 引入分层副本引用及 transfer 引用，保留 host-only 前缀 |
| `crates/infer-scheduler/src/infrastructure/kv_cache/radix_tree.rs` | 重导出 core RadixTree | 注意 core 树也供 worker beam search 使用，避免强迫 beam 路径依赖分层 I/O |
| `crates/infer-scheduler/src/application/kv_reclaim.rs` | 淘汰索引、释放预算、发送 `FreeKvIndices` | 区分丢弃、GPU 副本淘汰和待搬运；异步释放不能提前算为可分配 |
| `crates/infer-worker/src/domain/global_kv_alloc.rs` | worker 拥有 token-slot allocator | 回载使用新 slots，GPU 地址不需要与备份前一致 |
| `crates/infer-core/src/kv.rs` | 各层独立 K/V Tensor | 对离散 slots gather/scatter，合并连续区间 |
| `crates/infer-protocol/src/scheduler_to_worker_control.rs` | 有 free/preempt，当前无分层 transfer 协议 | 增加备份、回载、释放 host 副本及完成/失败事件 |
| `crates/infer-backend-cuda/src/config.rs` | 已有异步 D2H 与 pin host i32 能力 | 新建适合 KV 的有界字节池和 RAII 生命周期，不能直接照搬永久 pin 的 ABC helper |
| `crates/infer-worker/src/application/serve_loop.rs:242` | recurrent attention 禁用 KV-only prefix cache | 第一版排除 hybrid；后续同时恢复 recurrent/conv 状态 |

`domain/cache/snapshot.rs` 已有 recurrent checkpoint，但不能仅凭存在该类型，就认定已具备跨请求前缀复用能力。DeepSeek V4 自带压缩状态也要单独定义序列化语义。

特别注意：这里的 RadixTree 和 allocator 实际按 token slot 管理。传输 chunk 大小与 attention page 大小应分离，可以从 64/128/256 tokens 做实验，不能机械改成 SGLang 示例的 page size。

## 第一版设计建议

范围：单机 TP=1、普通 full attention、保持 KV 原精度、GPU/CPU 两层、只复用完全相同计算语义的连续前缀。存储 L3、跨实例复用、hybrid、TP、多模态和低比特压缩留给后续。

1. **职责划分**：scheduler 决定备份、命中和淘汰；worker 拥有 GPU/host 池并执行搬运。ZMQ 只传索引和事件，不传 KV 大数组。core 定义不含 CUDA 类型的 transfer port，CUDA adapter 实现。
2. **副本状态**：GPU 和 host 用独立可选引用；transfer 另外记录方向、ID、generation、源和目的 pin。一个互斥的 `Gpu/Cpu/Moving` 枚举不足以表达双副本。树的 owner 计数与 I/O 引用分开。
3. **备份策略**：先做有界、异步 write-through，在已提交的不可变前缀 chunk 上备份；只备份新增部分。可以先以请求完成为触发点跑通，再扩展到 prefill chunk 提交。备份队列满时跳过新备份，不阻塞 decode；显存紧急不足时允许按旧策略丢弃尚未备份的空闲前缀。
4. **GPU 淘汰**：有可用 host 副本时只释放 GPU slots，保留树；没有副本且未搬运时可普通淘汰。D2H 未结束不能释放源 slots。避免到显存耗尽才启动大批 write-back。
5. **CPU 回载**：连续前缀命中后预留 GPU 预算，由 worker 分配目标 slots，发起 H2D。请求处于等待回载状态，不计入 ready batch；整个目标前缀完成后再发布新索引。第一版不做逐层加载与计算交叠。
6. **搬运实现**：有界 pinned pool + 独立 copy stream + event 依赖。先合并物理连续区间；碎片严重时用预分配 staging 做 gather→DMA→scatter。测量 gather/scatter 和总线成本后，再考虑 mapped-host kernel 或布局转换。
7. **图执行**：保持 GPU KV pool 基址稳定，回载仅修改 slot 内容与索引；搬运在 graph replay 外完成。KV 生产完成→D2H、H2D 完成→计算都必须有明确依赖。
8. **host 淘汰**：有独立容量与引用计数；优先可回收叶部。不能留下不可恢复的前缀空洞。只剩 host 的祖先被移除时，需要同步截断或淘汰依赖它的后缀元数据。

第一版建议用受保护的节点/范围句柄处理备份；节点 split 时同步切分副本范围，或者明确禁止被搬运范围 split 并退回较短可用前缀。不能保存裸 NodeId 后假定回执抵达时节点仍表示原范围。

正确性约束：

- 每个 transfer 必须携带模型实例 epoch、对象 generation、操作 ID；重复和过期回执不能重复 free 或发布旧索引。
- 取消等待请求后，已经提交的 DMA 仍可能继续；等完成后回收资源，不能立即复用目的缓冲。
- 多个请求命中相同 host 前缀时合并回载，维护 waiter 引用。
- host 分配失败、回载失败可回退重算，但正在运行的搬运仍须收尾。
- 相同前缀不能跨模型版本/adapter/输入语义复用；未来 L3 key 要包含 namespace、前缀链 hash 和 KV 格式版本。
- TP 后续必须汇聚所有 rank 的完成状态，不能单 rank 完成就发布可用；collective 顺序必须一致。

## 如何判断值不值得开

普通 MHA/GQA、K/V head dim 相同时，单 token 的 KV 数据量为：

`bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element`

示例仅作估算：32 层、8 KV heads、128 head dim、BF16，为 128 KiB/token；8192 tokens 为 1 GiB。若实测有效 H2D 带宽为 25 GiB/s，纯搬运下限约 40 ms，另加排队、打包、同步。TP 要按本 rank 实际持有的 KV heads 计算；hybrid/MLA 不直接套这个公式。

收益条件是 `排队 + 打包 + 搬运 + 同步 < 重算该前缀的 prefill 时间`，还要计入后台 D2H 对 decode 的干扰。小模型、短前缀、低复用或总线繁忙时，CPU 命中未必更快。

## 验收与实施顺序

第一步：CPU 参考路径验证副本生命周期、备份后淘汰、重新分配 slots 回载、前缀分叉和预算守恒。

第二步：CUDA 有界池和异步搬运；逐层逐 slot 比较 round-trip KV 字节；随后与无缓存重算比较 logits/生成，数值容差单独定义。

第三步：接入调度等待态、取消、错误、重复回执、节点 split、并发共享前缀及内存压力。开启/关闭 CUDA Graph 各验证一次。

第四步：在 `bench/bench_prefix_cache.py` 基础上添加超过 GPU cache 容量、但可放入 CPU cache 的前缀工作集。比较关闭 prefix cache、GPU-only、GPU+CPU 三组；测试冷启动、GPU 热命中、CPU 热命中、无复用、前缀分叉和多轮对话。

记录 token 级 L1/L2 hit、实际跳过的 prefill tokens、D2H/H2D 字节与延迟、回载等待、host/GPU 占用、TTFT p50/p95/p99、TPOT/ITL p99、吞吐及失败回退数。GPU 预分配池下不要只看 nvidia-smi 的进程显存，要看可用 slots 和可容纳的 workload。

建议顺序：**两层缓存正确性 → 异步性能 → L3 file/backend 接口 → 按需求选择远程缓存或 FP8 → 低比特/淘汰实验**。FP8 与 L3 的先后应由活跃上下文容量压力、前缀复用率及多实例需求决定。
