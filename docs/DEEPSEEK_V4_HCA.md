# V4 HCA：压缩器与联合注意力 GPU kernel

这一步接在 [SWA kernel](DEEPSEEK_V4_SWA.md) 后面。实现单请求的 HCA
压缩器、decode 和 Tensor Core prefill，支持不等长分块与 CUDA Graph 重放。
固定压缩比 128、局部窗口 128、共享 KV 维度 512、压缩 KV 尾部 RoPE 维度 64，
支持 1–128 个 Query 头。在 16GB RTX 4070 Ti SUPER 上开发和验证，无需模型权重。

## 先理解数据流

```text
hidden states --wkv GEMM----> values (FP32) ─┐
              --wgate GEMM-> gates  (FP32) ─┼─ HCA compressor ─> compressed KV pool
                            learned ape ──┘                         │
query + raw KV ─────────────────────────────────────────────────────┤
                                                                  ↓
                           最近 128 个 raw KV + 所有已完成的压缩 KV
                                                                  ↓
                                          同一个 softmax + attention sink
                                                                  ↓
                                        output（后续需逆 RoPE、输出投影）
```

每个完整的 128-token 块产生一个 512 维向量，压缩的是 token 轴。
对块内 token `i` 和通道 `d`：

```text
score[i,d] = gates[i,d] + ape[i,d]
pooled[d]  = sum_i softmax_i(score[:,d])[i] * values[i,d]
compressed = partial_RoPE(RMSNorm(pooled), position = block_id * 128)
```

`values` 和 `gates` 是两个投影的结果；压缩器接口从这些结果开始，GEMM 不在计时内。
RoPE 使用块起点，且只旋转最后 64 个通道的相邻偶奇对。

位置 `t` 从零开始，其可见压缩块满足 `block_id < (t+1)/128`（整数除法）。
因此 `t=126` 没有压缩块，`t=127` 首次看见块 0。局部原始窗口与压缩块可以
代表部分相同的源 token；实现保留两种表示，不做去重，也不执行 top-k。

## 压缩器为什么只需要 6 KiB 状态

官方教学实现保存未完成块的 values/gates。这里每个通道只保存三个 FP32 数：
最大值 `m`、相对最大值缩放的权重和 `z`、加权值和 `u`。
每来一个新 value `v` 和 score `s`：

```text
m_new = max(m, s)
a = exp(m - m_new)
b = exp(s - m_new)
z_new = a*z + b
u_new = a*u + b*v
pooled = u_new / z_new
```

初值为 `m=-inf, z=0, u=0`。每步只更新 `3×512×4 = 6144 bytes` 的状态，
不会在第 128 步重新读取整块的两份 FP32 投影。两个指数中至少一个是 1，
kernel 只计算另一个指数。每个 CTA 256 线程，每个线程负责两个通道。

- `N=1`：一个 kernel 更新状态；恰好满块时融合 RMSNorm、RoPE 和压缩池写入。
- `N>1`：一个 kernel 并行计算所有完成块；第二个 kernel 更新最后未完成块的状态。
  前者只读旧状态，避免首块读取旧前缀时被尾块覆盖。
- 计算使用 FP32；在 pooling 后、RMSNorm 后、RoPE 后分别舍入到 BF16。
  这保留了未量化 BF16 参考路径的精度边界，不包含官方 FP8/QAT 模拟。

## 三个入口及调用顺序

接口位于 `crates/infer-core/src/ports/fused_ops.rs`，CUDA 实现位于
`crates/infer-backend-cuda/src/kernels/v4_hca/`。

| 入口 | 工作 |
|---|---|
| `FusedOps::v4_hca_compress` | 完整/分块 prefill 或逐 token 压缩，更新状态和压缩池 |
| `FusedOps::v4_hca_prefill` | 当前 chunk 的局部窗口与压缩池联合注意力，随后提交局部缓存 |
| `FusedOps::v4_hca_decode` | 单 token 联合注意力，融合局部环形缓存写入 |

先在同一 stream 调用 `compress`，再调用对应的注意力入口。Prefill 可以先生成
整个 chunk 的压缩 KV，注意力按每个 Query 的位置过滤尚不可见的块。

压缩器参数：

| 张量 | 类型、形状 | 含义 |
|---|---|---|
| values / gates | FP32 `[N,512]` | 投影结果；N 至少为 1 |
| ape | FP32 `[128,512]` | 学习得到的块内位置偏置 |
| norm | FP32 `[512]` | RMSNorm 权重；eps 单独传入 |
| rope | FP32 `[C,32,2]` | 每个块起点的 cos/sin；由调用者按模型 YaRN 配置准备 |
| start | GPU I32 `[1]` | 当前 chunk 起点或 decode 位置，不会自增 |
| state | FP32 `[3,512]` | 可写，依次为 m、z、u |
| compressed | BF16 `[C,512]` | 可写，绝对块编号索引的压缩池 |

注意力 Q/KV 已完成各自的归一化和 RoPE：

| 张量 | Prefill | Decode |
|---|---|---|
| query / output | BF16 `[N,H,512]` | BF16 `[H,512]` |
| new_kv | BF16 `[N,512]` | BF16 `[512]` |
| sink | FP32 `[H]` | FP32 `[H]` |
| start / position | GPU I32 `[1]` | GPU I32 `[1]` |
| compressed | 只读 BF16 `[C,512]` | 只读 BF16 `[C,512]` |
| local cache | 可写 BF16 `[128,512]` | 可写 BF16 `[128,512]` |

K 与 V 共用一个向量；分数缩放为 `1/sqrt(512)`。Sink 只进入共同的分母一次，
`-inf` 表示禁用。注意力 output 尚未执行逆 RoPE 和输出投影。

## 联合注意力的实现

Decode 每个 Query 头对应一个 256-thread CTA。先处理局部窗口，再以 128 行为单位
扫描可见压缩池，更新同一个 online softmax 的最大值、分母和输出累计量。
各段出现更大分数时，先重缩放已有累计量再合并，最后只归一化一次。
不能分别做两个 softmax 后直接把结果相加。

Prefill 每个 CTA 处理一个 token 的 16 个头，4 个 warp 用 BF16 WMMA Tensor Core
计算 QK 和 PV。Softmax 与输出累计量使用 FP32；概率拆成 BF16 高位与残差两部分，
通过两次 PV 乘加减小概率舍入误差。局部窗口与压缩池同样共用累计量。
WMMA fragment 的元素布局不公开，因此通过 shared memory 的行布局做重缩放，
不依赖未文档化的 lane 映射。第二个 kernel 在全部注意力读取结束后提交环形缓存。

这两条路径都不会创建随上下文增长的全局分数矩阵或临时显存。
Decode 只有第零个头提交当前 raw KV；所有头直接从 new_kv 读取当前槽位，避免跨 CTA 竞争。

## 状态与边界契约

- 张量连续、至少四字节对齐，位于 scope 的设备；可写参数不得与其他参数重叠。
- 请求独占自己的 state、压缩池和局部缓存，按单一 stream 顺序调用；张量须存活到执行完成。
- 从零开始时 state 和未使用缓存可以未初始化。块起点忽略旧 state；从块内位置继续
  时必须有正确的前缀累计量。满块后 state 归位，压缩池的历史行保留。
- `C` 表示完成块容量。负 start、末位置超过 I32 上限、完成块数超过 C 时，
  压缩器不改写 state/池；注意力输出 NaN，局部缓存不变。形状、alias 和 eps 错误由 Rust 返回错误。
- 设备位置在执行时读取，支持 Graph 动态更新；不会为了验证位置下载数据或同步 CPU。

## 验证与性能

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo test -p infer-backend-cuda --test v4_hca --test v4_swa \
  -- --ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo run -p infer-backend-cuda --example v4_hca_bench
```

HCA 测试使用独立 FP64 dense pooling/attention 参考，覆盖完整输入、不等长分块、
逐 token 压缩、prefill→decode、64 头、约 32K 历史、多次 online-softmax 重缩放、
第 128 个 token 的可见性、极端 gates/sink/logits、无效位置/容量、alias/stride，
以及更新输入和位置的 CUDA Graph。未使用缓存填 NaN；检查分配器计数不变。
压缩器的完整/分块/逐 token 路径在测试数据上结果逐位一致。
5 项 HCA 测试与原有 6 项 SWA 测试全部通过。
Rustfmt、diff 空白检查及 Clippy 通过；Clippy 沿用对已有 `dead_code`、
`extra_unused_type_parameters`、`manual_is_multiple_of` 告警的排除。

2026-09-17，RTX 4070 Ti SUPER（16GB，sm_89），64 个 Query 头，CUDA `-O3`：

| Prompt 长度 | HCA 并行 prefill（含压缩与缓存写入） | 逐 token HCA Graph | 加速比 |
|---|---:|---:|---:|
| 128 | 290.202 µs | 939.565 µs | 3.24× |
| 512 | 1478.246 µs | 6092.186 µs | 4.12× |
| 1024 | 2908.570 µs | 13097.779 µs | 4.50× |

全输出与逐 token 路径最大绝对差 `0.00390625`。两个比较路径都包括压缩和注意力，
都使用 Graph。CUDA events 测七组、每组五次重放，取每次调用时间的中位数。

| 已有历史 token | 可见压缩行 | Decode 联合注意力 |
|---|---:|---:|
| 4096 | 32 | 12.017 µs |
| 32768 | 256 | 29.190 µs |
| 131072 | 1024 | 79.980 µs |

Decode 使用 32 次调用组成的 Graph 摊薄提交空隙，位置固定在块起点，KV 使用可重复的
固定快照。压缩器在该不满块场景单独测得约 0.95 µs；该数字不含每 128 步的满块输出成本。
这些结果均不含投影 GEMM、传输、逆 RoPE、输出投影或模型其他部分，不代表完整模型吞吐。
原始结果在本地 `target/v4-hca-bench.jsonl`。

ptxas：压缩 decode 27 寄存器/线程、完成块 38、尾部状态更新 40；注意力 decode 63，
prefill 126。注意力 shared memory 分别为 16,952 和 46,272 bytes，所有 kernel 零 spill。
当前 WSL/WDDM 调试接口不能初始化 Compute Sanitizer，因此没有 sanitizer 通过结论。

## 后续边界

这些入口是独立 GPU 算子，尚未接入正式 V4 模型执行器、请求调度和权重加载。
完整权重适配还需要 FP8/QAT 等精度路径。当前 decode 每个头顺序扫描压缩池，
长历史的后续优化方向是 Split-KV 与跨头 KV 复用，应按测量结果逐项推进。

参考：[官方 compressor / attention](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)、
[Transformers 架构与可见性规则](https://huggingface.co/docs/transformers/model_doc/deepseek_v4#attention-mask-layout)。
