# KV Cache 压缩算法选型

日期：2026-09-19。用户当前优先级改为 KV 压缩，HiCache 暂后置。本文依据论文、作者仓库、引擎文档及本地代码做选型；尚未实现压缩或运行复现 benchmark。上游 main/在线文档随时间变化。

## 推荐

- 若首要目标是 H100/H200 上的在线吞吐：先做 **FP8 KV**，并接通硬件适配的 attention backend。
- 若首要目标是 16GB GPU 的长上下文容量、希望超过 2 倍压缩：优先做 **KIVI 风格 INT4 基线**，随后对比 **KVarN**。这是一条实现顺序建议，不是声称 KIVI 质量优于 KVarN。
- 研究候选排序：KVarN（长推理误差与实用实现）和 TurboQuant 4-bit（旋转量化代表）优先；2-bit、跨层预测、token 淘汰放在质量评测之后。
- 不存在与硬件、模型、batch、上下文无关的“最好”。比较应同时约束精度、常驻字节、临时 workspace、TTFT、decode 延迟和服务吞吐。

本机 `nvidia-smi` 显示 RTX 4070 Ti SUPER 16GB；仓库 README 的性能基准是 H200。两者不能套用相同的加速结论。cuDNN 当前文档支持矩阵中，Ampere/Ada 的 SDPA 列出 FP16/BF16，而 Hopper 列出 FP8。因此 Ada 上支持 FP8 数值类型，不代表本项目能直接使用 cuDNN FP8 paged attention。[cuDNN 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/v1.25.0/operations/Attention.html)

## 候选比较

以下压缩比均以 16-bit KV 数据为参照；理论位宽比例不是整个推理进程的显存比例。

| 方案 | 核心方法 | 压缩能力/代价 | 本项目定位 |
| --- | --- | --- | --- |
| FP8 | E4M3/E5M2 + scale | 数据主体约 2×；质量与速度依赖 scale 和 backend | 生产基线、Hopper 优先 |
| 分组 INT4 / KIVI | K 按 channel、V 按 token，保留高精度尾部 | 4-bit 主体理论 4×，实际较低；分组管理与解包成本 | 低比特第一条可控实现 |
| KVarN | Hadamard + token/channel 双轴方差归一化 + RTN | 额外 scale 与 tile 编码；作者有服务实现 | 新算法重点候选 |
| TurboQuant | 旋转、非均匀量化与误差处理；实现存在变体 | 低比特能力强，但解码与质量取舍明显 | 4-bit 对照实验 |
| KVQuant | pre-RoPE K、非均匀码本、离群值独立保留 | 需要校准与稀疏/稠密读取组合 | 高压缩精度研究 |
| AQUA-KV | 跨层/KV 预测，只量化残差 | 需要校准 predictor，增加重建依赖 | 暂不作为首版 |
| SnapKV/PyramidKV | 选择 token、分配层预算 | 不可逆丢弃历史；索引与质量复杂 | 单独 opt-in 功能 |
| KVzip | 面向未来不同 query 的 KV 选择 | 有额外分析/重建成本，仍为有损淘汰 | 多 query 复用研究 |
| MiniCache | 合并相邻层的相似 KV | 跨层状态与重建，可能叠加量化 | 暂不作为首版 |

## 1. FP8：最稳的工程起点，但要区分存储与计算

可以使用原精度 Q/O、FP8 KV 存储，在 attention tile 内反量化；也可以使用受支持的 FP8 attention 运算路径。两者不是同一种性能模型。scale 粒度必须与目标 kernel 匹配：不能自行选择 per-token scale 却假定只支持 per-tensor/per-head scale 的 backend 可以直接消费。

建议 E4M3 作为候选，保留 E5M2 对照，做代表性长上下文校准与异常值统计，避免默认 scale=1 被当作可靠质量策略。允许敏感层保持 BF16。不要把 FP8 权重加载支持误认为 FP8 KV 支持。[vLLM 文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

## 2. KIVI：适合学习和建立低比特基线

KIVI 的关键是 K 与 V 分布不同：K 用 per-channel、V 用 per-token 量化，并保留高精度 residual cache 处理增量生成。作者提供 2/4-bit 实现。[论文](https://arxiv.org/abs/2402.02750)、[代码](https://github.com/jy-yuan/KIVI)

建议先以 K4V4 验证，质量通过后再试 K4V2。工程难点是 K 的量化组跨多个 token：tail 尚未填满时保持高精度，组完成后冻结编码。共享前缀的 tail 不可就地修改；需 copy-on-write 或只复用已封存组。

以每 64 个值共享 FP16 scale 和 zero-point 的普通 4-bit affine 编码为例，平均位宽为 `4 + (16+16)/64 = 4.5 bit`，主体压缩约 `16/4.5 = 3.56×`。这只是存储公式示例，并非 KIVI 整体实测；tail、padding、索引及 workspace 会进一步降低收益。

## 3. KVarN：最值得深入复现的新候选

KVarN（2026-06）关注真正 autoregressive decode 中的误差累积：先做 channel 方向 Hadamard，再对 token/channel 双轴做方差归一化，最后量化。论文质量评估覆盖生成任务，强调 token scale 误差。[论文](https://arxiv.org/abs/2606.03458)

作者仓库提供 vLLM fork 和 Triton kernel，公开预设为 K4V2、64/128-token tile，并报告若干模型配置下数倍 KV 容量及吞吐收益。这些是作者结果，不能视为 RustInfer 或本机的复现。密集路径文档提到 FP16 compute；移植 BF16 模型时要显式验证 dtype 转换。仓库也承认固定 workspace 对紧张单卡预算的影响。[作者仓库](https://github.com/huawei-csl/KVarN)

比起直接移植整个 vLLM backend，更适合先实现独立 codec，与 KIVI 风格 INT4 在相同有效位宽下比较 attention 输出与长生成质量，再决定是否移植高性能读取路径。

## 4. TurboQuant：值得关注，但不应依据宣传数字直接选定

论文使用随机旋转与量化，并讨论内积误差校正；社区和引擎中的 norm-correction、K8V4、4bit 等预设不是完全相同的编码。复现必须固定具体变体。[论文](https://arxiv.org/abs/2504.19874)

vLLM 2026-05-11 在 v0.20.2 上比较 BF16、FP8 和四种 TurboQuant，覆盖 H100、多模型、长检索和推理。结论是 FP8 更适合作为默认；4bit-nc 可换取容量，而更激进变体出现更明显质量下降。部分低比特配置降低吞吐，显存受限时仍可能通过减少排队改善 TTFT。该报告不是对之后所有 kernel 的永久结论，但足以否定“位宽越低就越快”的假设。[实测报告](https://vllm.ai/blog/2026-05-11-turboquant)

## 5. KVQuant、AQUA-KV、淘汰与硬件 FP4

KVQuant 的 pre-RoPE 设计使读取路径需要处理位置旋转，加上非均匀码本、离群值与校准，明显增加当前 fused RoPE/scatter 路径的改造成本。[论文](https://arxiv.org/abs/2401.18079)

AQUA-KV 利用 predictor 压缩无法预测的残差，作者报告在所测 Llama 模型上低至 2–2.5 bit 的高质量结果，但需要模型相关校准，重建也有依赖。适合研究极限压缩，不适合先建立通用缓存接口。[论文](https://arxiv.org/abs/2501.19392)、[代码](https://github.com/goodevening13/aquakv)

SnapKV/PyramidKV 的 attention 观察与层/head 预算会改变历史表示；KVzip 专门考虑未来 query；MiniCache 在层维度合并表示。它们不只是替换存储 dtype，故工程范围大于简单量化。[SnapKV](https://arxiv.org/abs/2404.14469)、[PyramidKV](https://arxiv.org/abs/2406.02069)、[KVzip](https://arxiv.org/abs/2505.23416)、[MiniCache](https://arxiv.org/abs/2405.14366)

SGLang 已有实验性 FP4 KV 路线及 SM100 native recipe，但具体模型、head dim、page size 和硬件支持受限制，不能视为 H200/4070 上的直接替代品。INT4 编码也不等于硬件 NVFP4 attention。[SGLang 文档](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache)

## RustInfer 的实施边界

本地代码已经核实：

- `infer-core/src/kv.rs` 有 `KvQuantTier`，但 `PagedKvLayer<T,D>` 的 K/V 仍是 `Tensor<T,D>`；不能只给 enum 增加 INT4 就宣称完成压缩。
- `flash_attn_gqa/mod.rs::attention_paged` 的 Q、K、V、O 共用泛型 T；实际入口为 BF16/FP16。需要将存储格式与计算 dtype 解耦。
- decode 默认优先 cuDNN；短 query、ragged prefill、mixed batch 还有其他路径。所有读取 KV 的路径都必须审计，不能只改普通 decode。
- `components/attention_core.rs` 有 Qwen3 的 norm+RoPE+scatter 和 Llama 的 RoPE 后 scatter。选择 post-RoPE 量化可减少首版改造范围；KVQuant pre-RoPE 另行实现。
- token-slot allocator 与 radix prefix reuse 不天然等同于 64/128-token 量化组。需要独立逻辑量化组和物理存储管理，否则少量存活 token 可能 pin 住整个组。
- hybrid 模型的 full-attention 层可以单独量化，recurrent state 保持原格式；这与 hybrid 跨请求 prefix reuse 是否支持是不同问题。

建议公共接口表达：存储格式、K/V 编码与 scales、logical-to-physical 映射、sealed groups、原精度 tail，以及实际分配字节。decode 在 tile/register/shared-memory 内反量化并立即计算；避免长期保留全层全序列的 dense shadow。若暂用有界分层 workspace，必须把其成本单独计入报告。

分区 attention（压缩 body + 高精度 tail）需要合并 softmax 的全局归一化。不能分别得到两个 normalized output 后简单相加；每分区应输出 max、exp sum、weighted sum 或等价 LSE 供正确合并。

实施顺序：

1. 格式描述、容量计算、CPU codec 参考及误差统计；固定一个模型和 head dim 验证。
2. 根据目标硬件先接 FP8 或 K4V4；写入量化与 decode 读取均实现，保留 BF16 对照。
3. 补全 chunked prefill、prefix hit、mixed batch、CUDA Graph、共享尾组及异常回收路径。
4. 在相同有效内存预算下比较 KVarN、TurboQuant 4-bit；达到质量门槛后再降 V 或 K 位宽。

## 必须跑的验证

- **数值**：codec、量化 attention 与 CPU dequant reference 比较；零向量、极值、tail/group 边界、不同 head dim、不同 K/V scale。
- **语义**：分块 prefill 与一次 prefill、共享前缀分叉、slot 回收、图执行、混合 batch。允许量化误差，不要求与 BF16 token-by-token 完全一致。
- **质量**：真正读取量化历史的逐 token decode；长检索/多针、数学、代码、长生成、多轮切换。单次不读历史 KV 的 full-forward PPL 不能验证这条路径。
- **内存**：payload + scale/zero + tail + padding + 索引 + staging/workspace + graph 固定开销；区分每 token 缓存和整个进程显存。
- **性能**：短/长上下文与低/高并发，记录 prefill、TTFT、TPOT/ITL p99、吞吐；额外记录每次量化组封存引起的延迟峰值。

最终选择依据是目标 workload 的质量/容量/速度三项实测，而不是论文 headline 压缩倍数。
