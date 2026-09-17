# V4 Lightning Indexer 第一步：融合打分 kernel

接在 [CSA 重叠压缩器](DEEPSEEK_V4_CSA.md) 后，实现 Indexer 的打分阶段：
点积 → 每头 ReLU → 带权跨头求和 → 因果屏蔽。
入口为 `FusedOps::v4_indexer_scores`，CUDA 文件在
`crates/infer-backend-cuda/src/kernels/v4_indexer/`。
单请求、128 维索引向量、1–128 个索引头，Flash 使用 64 头。

打分入口输出 FP32 分数矩阵，可接 [top-k 算子](DEEPSEEK_V4_TOPK.md) 得到条目编号。
索引 Q/K 的投影、独立索引压缩器、RoPE、Hadamard/量化准备，以及稀疏主注意力和
模型执行器集成仍待实现。
已有 CSA 压缩器产生的是主注意力的 512 维 KV；Indexer 使用独立学习的 128 维 Key，
不能直接截取主 KV 的前 128 维替代。

## 原理与顺序

对当前 token t、压缩条目 j、索引头 h：

```text
dot[t,h,j]   = sum_d Q[t,h,d] * K[j,d]
score[t,j]   = sum_h weights[t,h] * max(dot[t,h,j], 0)
weights[t,h] = projected_weights[t,h] / sqrt(128 * H)
```

weights 从当前 hidden state 投影而来；传给此入口前已经包含缩放。
Q 则从 query 的低秩中间表示经过独立投影得到。
K 在所有索引头之间共享。

ReLU 必须放在每个头的点积之后。weights 可能为负，最终 score 也可能为负，
所以无效条目的分数必须是 `-inf`，不能填零。这里不做 softmax，后续只需按分数
排序选取条目；主注意力会用自己不同的 Query/KV 重新计算最终权重。

零基绝对位置 p 的可见条目满足 `j < floor((p+1)/4)`：

| token 位置 p | 可见压缩行 |
|---|---|
| 0、1、2 | 无，整行分数为 -inf |
| 3 | 0 |
| 4、5、6 | 0 |
| 7 | 0、1 |

每个 chunk 内的 Query 使用各自的位置。可以先准备整个 chunk 的压缩 Key，
再打分；kernel 会屏蔽对当前 Query 而言尚未完成的块，不读取这些 Key。

## 如何让 decode 也使用 Tensor Core

矩阵乘法的一维使用索引头，而不是只使用 token：

```text
Q 的一个头分块    [16 heads, 128 dims]
K 的一个条目分块  [16 keys,  128 dims]

Q × K^T → [16 heads, 16 keys] FP32
                       ↓ ReLU、带权沿 heads 求和
                   [16 keys] FP32
```

因此，即使 N=1，多个索引头仍提供足够的矩阵行。
一个 CTA 有 128 个线程、4 个 warp，负责一个 token 的 64 个候选 Key。
每个 warp 计算其中 16 个候选；每次处理 16 个头，遍历全部头分块。
Key 分块一次载入 shared memory 后在头分块间复用，Q 在四个 warp 之间复用。

H≤32 时一次缓存全部 Q 和权重。33–64 头且网格不超过 128 个 CTA 时，也一次
缓存全部 Q，只在开始做一次 block 同步，其余是 warp 内同步。更大的网格或
H>64 使用每次载入 16 个头的路径，以较小 shared memory 占用换取更多并发 CTA。
这个形状阈值来自当前 sm_89 开发卡上的测量，未来换 GPU 应重新测量；没有运行时
autotuning、位置读回或依赖设备 position 的主机分支。

WMMA 使用 BF16 输入、FP32 点积累计。通过文档规定的 row-major store 将 fragment
写到 shared memory 后执行 ReLU/头归约，不依赖未公开的 fragment lane 布局。
每个输出分数只有一个 CTA 写入，不使用 atomic，也不需要第二次归约 kernel。

整个入口只有一次 launch，无临时 GPU 分配、CPU 同步或设备位置读回。
中间点积不写到全局显存。以 N=1024、H=64、C=256 为例，FP32 `[N,H,C]`
中间矩阵原本需 64 MiB，这里仅保留 1 MiB 的 `[N,C]` 输出。
输出由调用者预分配；“无临时分配”不表示没有分数输出的显存成本。

## 接口契约

| 参数 | 类型、形状 | 含义 |
|---|---|---|
| query | BF16 `[N,H,128]` | 已准备好的索引 Query |
| keys | BF16 `[C,128]` | 独立索引压缩 Key 池，只读 |
| weights | FP32 `[N,H]` | 已包含 `1/sqrt(128*H)` 的权重，允许负数 |
| start | device I32 `[1]` | chunk 绝对起点或 decode 位置，只读、不自增 |
| output | FP32 `[N,C]` | 有效分数及未来位置的 -inf |

N=1 对应 decode，N>1 对应完整或分块 prefill；采用相同计算顺序。
要求 N,C>0，1≤H≤128；张量连续、至少四字节对齐，位于 scope 的设备；
output 不得与任何输入重叠。展平 CTA 数 `N*ceil(C/64)` 不得超过 CUDA grid.x 上限。
输入与输出须存活至异步执行完成，调用者须在同一 stream 上先发布有效 Key。

负 start、最后位置超过 I32 上限、完成块数超过 C 时，整个 output 填 NaN，
输入保持不变。维度、stride、对齐和 alias 错误由 Rust 立即返回错误。
有效 Q/K/weights 必须有限；未使用的 Key 行可填 NaN。
每次调用覆盖全部输出，包括未来位置，不需要外部清零；Graph 重放读取新的输入与 start。

本入口对准备好的 BF16 Q/K 做 FP32 累计，ReLU、乘权和头归约均为 FP32。
它没有复现教学 PyTorch 路径每个 BF16 中间张量的舍入，也没有内部执行 FP4/QAT。
完整量化模型的排序一致性需在后续准备 Q/K、接入 top-k 后另行验证。

## 测试与微基准

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test -p infer-backend-cuda --test v4_indexer --test v4_csa \
  --test v4_hca --test v4_swa \
  -- --ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo run -p infer-backend-cuda --example v4_indexer_bench
```

独立 FP64 参考使用标量点积和跨头求和，不复用 WMMA 分块。FP32 误差界按
`sum_h |weight[h]| * sum_d |q[h,d]*k[j,d]|` 计算，允许不同累计次序的误差，
也覆盖正负项抵消。解析测试直接验证 ReLU 顺序、负权重、零权重和因果屏蔽。

覆盖 1/7/16/17/63/64/65/128 头、候选 tile 尾部、输出前后 guard、未使用 Key
填 NaN、32K 历史、极端但有限的点积、无效位置/容量、stride、对齐和 alias。
完整、分块、逐 token 路径要求逐位相同；Graph 更新 Q/K/weights/start，检查
重启请求与显存分配器计数。当前 6 项 Indexer GPU 测试通过。

微基准使用 64 头，通过 CUDA events 测七组、每组五次 Graph 重放，取中位数。
Prefill 每个 Graph 包含 8 次完整调用；decode 包含 32 次，摊薄主机提交间隙。
两种计时不包含输入传输、投影、索引压缩、Q/K 变换、top-k 或主注意力。
Prefill 同时与本地逐 token Graph 路径比较，检查每个有效分数有限、mask 正确及
输出逐位一致；这些比值不是相对官方实现或整个模型的加速比。

2026-09-17，RTX 4070 Ti SUPER（16GB，sm_89），最终调度路径测得：

| Prefill token 数 | 起点 | Key 容量 | 并行打分 | 逐 token Graph |
|---|---:|---:|---:|---:|
| 128 | 0 | 32 | 6.067 µs | 552.141 µs |
| 512 | 0 | 128 | 26.650 µs | 2472.954 µs |
| 1024 | 0 | 256 | 77.542 µs | 4743.578 µs |
| 128 | 32768 | 8224 | 491.110 µs | 808.755 µs |

| Decode 时已到达 token 数（含当前） | 可见压缩 Key | 单次打分 |
|---|---:|---:|
| 128 | 32 | 4.320 µs |
| 1024 | 256 | 4.813 µs |
| 32768 | 8192 | 5.898 µs |
| 131072 | 32768 | 18.225 µs |
| 1048576 | 262144 | 165.694 µs |

Decode 表中容量等于可见 Key 数；并非在最大预留容量下测试所有历史长度。
原始结果在本地 `target/v4-indexer-bench.jsonl`。缓存全部 64 头的单一路径实测
会降低大网格吞吐，因此仅用于小网格；两种路径的数学累计次序保持相同，
跨调度路径的 prefill、chunk 和 decode 结果也逐位一致。

ptxas（sm_89、`-O3`）资源报告：

| 路径 | 寄存器/线程 | Shared memory |
|---|---:|---:|
| 缓存最多 16 头 | 38 | 27,200 bytes |
| 缓存最多 32 头 | 38 | 31,872 bytes |
| 缓存最多 64 头 | 38 | 41,216 bytes |
| 每次载入 16 头 | 56 | 27,200 bytes |

各路径均无 spill、无 stack frame。6 项 Indexer 测试与原有 17 项 SWA/HCA/CSA
回归测试全部通过；Rustfmt、diff 空白检查与 Clippy 通过。Clippy 沿用对已有
`dead_code`、`extra_unused_type_parameters`、`manual_is_multiple_of` 告警的排除。
本轮没有运行 Compute Sanitizer；此前环境的 WSL/WDDM 调试接口初始化失败。

## 后续性能边界

Indexer 仍扫描可见索引 Key，成本随历史长度增长。当前输出是稠密 `[N,C]`，
预留容量远大于已用容量时还要写入未来行的 -inf；这支持固定形状 Graph，
也意味着短历史的成本不完全由可见条目数决定。
当前 top-k 已与这个接口组合验证；后续可按测量考虑分块候选选择与打分融合，
减少完整分数矩阵的全局写入。当前没有实现这种融合或完整 CSA 执行链。

参考：[官方 Indexer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)、
[Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)。
