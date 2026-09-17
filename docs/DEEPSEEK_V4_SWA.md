# V4 SWA：GPU prefill 与 decode

入口是 `FusedOps::v4_swa_decode` 和 `FusedOps::v4_swa_prefill`，CUDA 实现在
`crates/infer-backend-cuda/src/kernels/v4_swa/`。
支持单请求的整段输入、分块输入与逐 token decode，包含环形缓存写入；不依赖模型权重。

## Decode 输入和输出

| 张量 | 类型 / 形状 | 含义 |
|---|---|---|
| `query` | BF16 `[H, 512]`，`1 <= H <= 128` | 已完成归一化和 RoPE 的当前 Query |
| `new_kv` | BF16 `[512]` | 当前 token 的共享 KV，已完成归一化和 RoPE |
| `sink` | FP32 `[H]` | 每个头的额外 softmax logit；`-inf` 可禁用 |
| `position` | GPU 上的 I32 `[1]` | 从零开始的绝对位置，算子不会自增 |
| `cache` | BF16 `[128, 512]` | 请求独有的环形缓存，原地更新 |
| `output` | BF16 `[H, 512]` | 注意力输出，尚未执行逆 RoPE 和输出投影 |

所有张量连续、至少四字节对齐，位于执行 scope 的 GPU 上。
可写的 cache/output 不得与其他参数重叠。它们必须存活到异步执行完成。
从位置零开始调用时，未使用的缓存槽位不要求清零。
如果从中间位置开始，调用者必须预先填好该请求的窗口历史。
同一缓存按 stream 顺序调用，不允许两个请求或两个 stream 并发改写。
负位置使输出变成 NaN、缓存保持不变；执行路径不为检查设备值而下载张量。

## 计算原理

位置 `t` 只读取 `max(0,t-127)..=t`。每个头计算：

```text
score[j] = dot(query, kv[j]) / sqrt(512)
denom    = exp(sink) + sum_j exp(score[j])
output   = sum_j exp(score[j]) * kv[j] / denom
```

实现先减去包含 sink 在内的最大分数，再求指数，避免溢出。
sink 只贡献分母，不对应一个真实的历史 token，也不贡献 value。
Decode 的 QK、softmax 和加权求和使用 FP32，最后一次转换为 BF16。
这定义的是 BF16 输入下的注意力数学计算，不包含官方权重的 FP8/QAT 模拟。

缓存位置是 `t % 128`，但 RoPE 使用的仍是绝对位置 `t`。
窗口满后，全部 128 个物理槽位有效，注意力求和不要求按时间重新排序。
因此无需搬动 KV，也无需维护每个 token 的索引表。

## Decode 的 CUDA 实现

一次调用只启动一个 kernel，且不申请显存、不进行 CPU 同步或数据回传。
每个 Query 头对应一个 256-thread CTA，8 个 warp 分摊窗口内的 token：

1. 用 BF16 pair 向量化读取 Q/KV，FP32 点积，并在 warp 内归约。
2. 在 shared memory 保存最多 128 个分数，融合稳定 softmax 与 sink。
3. 各 warp 计算部分加权和，在 shared memory 合并后写入输出。
4. 只有第零个头的 CTA 写入新缓存行。

第 4 步没有跨 CTA barrier。正确性依赖一个明确规则：**所有 CTA 遇到当前
槽位时，直接读 `new_kv`，从不读即将覆盖的缓存行**。这样缓存写入不会与别的
头发生读写竞争，也省掉一次单独的 scatter kernel。

这是为短窗口 decode 实现的 CUDA SIMT 内核，尚未使用 Tensor Core。
KV 跨头复用依靠 L2 缓存；后续可基于测量评估多头分组、Tensor Core 等优化。
位置保存在 GPU 张量中，因此 CUDA Graph 重放可以使用更新后的位置而无需重新捕获。

## Prefill：并行处理多个输入位置

`v4_swa_prefill` 的 Q/output 形状为 `[N,H,512]`，新 KV 为 `[N,512]`，
sink 和 cache 与 decode 相同。`start_position` 是 GPU 上的 I32 `[1]`，
表示本段第一个 token 的绝对位置，算子不会自增。整段提示词从零开始；
分块输入依次传入各块的起点，使用同一个缓存。N 必须为正且不超过 I32 上限。
每个位置仍然只能读取自己及前面最多 127 个 token，不会读取未来 token。

一次调用有两个按同一 stream 排序的 kernel：

1. **计算注意力**：旧 cache 全程只读。Query 窗口中早于本块的位置从环形
   cache 读取，本块内的位置从新 KV 张量读取。
2. **提交缓存**：注意力 kernel 完成后，将本块最后 `min(N,128)` 行写入
   `(start_position + row) % 128`。短块保留其余历史槽位。

例如已有 128 个历史 token，新块有 200 个 token。新块最前面的 Query 仍需
读取旧历史，因此不能先把这 200 行批量覆盖到 128 槽的缓存里。
两个 kernel 之间的 stream 顺序保证历史读完后才被覆盖，不需要 CPU 同步或临时缓存。
Prefill 结束后直接使用同一份 cache，以 `start_position+N` 调用 decode 即可。
负起点或末位置超过 I32 上限时，输出 NaN，cache 不变。

计算 kernel 每个 CTA 负责 **一个 token 的 16 个 Query 头**，用 4 个 warp
和 BF16 WMMA Tensor Core 计算 QK 与 PV。尾部不足 16 个头时内部补零，输出只写有效头。
KV 每次加载 16 行，在这些 Query 头之间复用。窗口内分数仅保存在 shared memory，
不会创建全局 `N×N` 注意力矩阵。每个 CTA 使用 45 KiB shared memory。

归一化在 FP32 中计算，包含每个头的 sink。为减少把概率直接转换到 BF16 的误差，
概率表示成两个 BF16 分量：`hi=BF16(p)`、`lo=BF16(p-float(hi))`，
分别计算 `hi×V` 和 `lo×V`，用 FP32 累加。最终输出为 BF16。
因此 prefill 与 SIMT decode 使用不同的浮点运算次序，不要求逐位相等。
同一输入的整段/分块 prefill 使用相同运算次序，测试要求输出逐位一致。

两个 kernel 均无临时 GPU 分配，支持固定形状的 CUDA Graph 捕获；重放时可更新
起点和输入张量内容。调用者负责保证请求历史、stream 顺序和张量生命周期有效。

## 验证和测量

在 RTX 4070 Ti SUPER 上使用 `sm_89`：

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  cargo test -p infer-backend-cuda --test v4_swa --no-run

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo test -p infer-backend-cuda --test v4_swa -- --ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo run -p infer-backend-cuda --example v4_swa_bench

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo run -p infer-backend-cuda --example v4_swa_prefill_bench
```

测试使用独立的 FP64、按时间顺序的 dense attention 作为参考，不复用环形索引逻辑。
覆盖 64 个 Query 头、窗口未满/刚满、两次覆盖、交错请求、极端 sink/logit、
大绝对位置、CUDA Graph 动态输入、非法形状和重叠内存；未使用槽位填 NaN。
同时检查调用前后分配器计数，确认没有临时 GPU 分配。

Prefill 测试另外覆盖：273-token 整段与不等长分块、17 个头的尾部、prefill→decode，
64 头的 Graph 动态输入、因果性、末位置溢出。数值参考是独立 FP64 attention，
每元素阈值为 `1e-5 + 0.004 * abs(reference)`，包括 BF16 输出舍入误差。

基准用 CUDA events 测量七组样本的中位数，输出三种时间：

- `eager_stream`：普通 launch，包含 CPU 提交不及时造成的 stream 空隙。
- `single_node_graph_stream`：单节点 graph 重放，也可能包含提交空隙。
- `bundled_graph_device`：一个 graph 包含 32 次算子调用，摊薄提交开销。

基准固定满窗口，重复相同输入和位置，不计输入上传、模型投影、采样或调度，
因此结果是算子微基准，不能换算成完整模型的 tokens/s。

2026-09-16，RTX 4070 Ti SUPER（sm_89），CUDA 编译优化 `-O3`，本机结果：

| 测量方式 | 每次调用中位数 |
|---|---:|
| 普通 launch 的 stream 时间 | 11.436 µs |
| 单节点 Graph 的 stream 时间 | 11.088 µs |
| 32 节点 Graph 摊销后的设备时间 | 9.439 µs |

三项 GPU 测试均通过。ptxas 报告每线程 40 个寄存器、每 CTA 16,948 bytes
shared memory、零寄存器 spill。当前 WSL/WDDM 环境缺少可用的调试接口，
Compute Sanitizer 初始化失败，因此尚未完成 sanitizer 内存检查。

Prefill 基准将两个路径都捕获为 Graph：一次并行 prefill（含缓存提交），
以及用相同 Q/KV 顺序执行 N 次 decode。使用 CUDA events 测量七组样本中位数；
计时不含输入上传和模型其他算子。计时外检查所有输出及无临时 GPU 分配。
这个比较只衡量本仓库两个 SWA 路径，不代表与其他推理库或完整模型的性能比较。

2026-09-17，同一张 RTX 4070 Ti SUPER，BF16、64 个 Query 头、W=128、D=512：

| 输入 token 数 | Prefill（含缓存提交） | 逐 token decode Graph | 加速比 |
|---|---:|---:|---:|
| 128 | 230.771 µs | 805.216 µs | 3.49× |
| 512 | 1178.816 µs | 4827.124 µs | 4.09× |
| 1024 | 2278.810 µs | 9564.979 µs | 4.20× |

所有输出与逐 token decode 的最大绝对差为 `0.00390625`。
六项 GPU 测试全部通过（包含原来的三项 decode 测试）。Prefill 的 ptxas 报告为
每线程 124 个寄存器、每 CTA 46,080 bytes shared memory、零寄存器 spill。
后续公共 tile 代码抽取后为 125 个寄存器，shared memory 不变，仍零 spill；
维护回归数据见 [top-k 维护验证](DEEPSEEK_V4_TOPK.md#自动回归与公共实现)。
Clippy 检查通过，运行时排除了已有的 `dead_code`、`extra_unused_type_parameters`
和 `manual_is_multiple_of` 告警；未改动这些无关的原有代码。

## 下一步

接入模型的归一化/RoPE、逆 RoPE、分组输出投影和正式请求缓存管理。

参考：[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)、
[官方注意力实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)、
[NVIDIA WMMA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。
