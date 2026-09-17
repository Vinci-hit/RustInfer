# V4 Lightning Indexer 第二步：确定性 top-k

接在 [Indexer 打分 kernel](DEEPSEEK_V4_INDEXER.md) 后，从每个 Query 的 FP32 分数
中选择最多 K 个压缩条目。支持 K=1–512，Flash 使用 K=512。
入口为 `FusedOps::v4_indexer_topk`，CUDA 实现在
`crates/infer-backend-cuda/src/kernels/v4_indexer/v4_indexer_topk.cu`。

```text
准备好的索引 Q/K、头权重
       ↓ v4_indexer_scores
scores [N,C]，FP32
       ↓ v4_indexer_topk
indices [N,K]，I32
       ↓ 后续稀疏主注意力
读取这些编号的主压缩 KV + 最近 128 个原始 KV
```

现在打分与筛选两阶段可以在同一 stream/Graph 中串联。完整模型的独立索引 Q/K
准备、Hadamard/量化、稀疏主注意力及模型执行器尚未接入；这里也没有融合打分与筛选。

## 选择规则

输出按“分数降序，同分时压缩编号升序”排列，有效编号不重复。
`+0` 与 `-0` 视为同分。NaN、`-inf` 不参与选择，`+inf` 排在所有有限分数之前。
有效候选不足 K 时，尾部补 `-1`；没有候选时整行都是 `-1`。
这是一套明确、可重放的规则，不承诺复现 PyTorch `topk` 未指定的同分顺序。

例如 K=4：

```text
编号       0      1       2      3      4
分数     -3.0    8.0    -inf    8.0    NaN
输出       [1, 3, 0, -1]
```

算子重新检查因果可见性：位置 p=start+t 只读取 `j < (p+1)/4` 的分数。
即使未来条目被误填成很大值，也不会被选中。位置 0–2 的输出全为 `-1`，位置 3
首次允许编号 0。候选不足时不能把填充编号当作合法 KV 读取，后续注意力应跳过 `-1`。

返回值是压缩池里的绝对行号，未加上原始 KV 窗口的偏移。主注意力使用独立的
压缩池指针即可直接按行号读取；若以后拼成一个 raw/compressed 大数组，偏移应由
对应注意力布局处理。

## 分块选择为什么不会漏掉全局 top-k

将一行按 2048 个分数分块，各块独立选择自己的 top-k。
一个条目若在本块已排到 K 名以后，那么同块至少有 K 个条目比它优先，
它不可能进入全局 top-k。因此先丢弃这些局部落选条目是精确的，无需近似阈值。

每块由 256 个线程处理，使用 CUB 的稳定 32-bit 浮点 radix sort；输入按编号升序
排列，稳定排序自然保持同分时较小编号优先。先屏蔽无效值、规范化零的符号，
再排序。这里只排序固定大小的块，不对整个历史做全局排序。

C≤2048 时，每行一个 CTA 直接输出结果。根据 C 与 K 的较大值选择
256/512/1024/2048 个槽位，容纳不足 K 时的填充。更长的行先将每块 top-k 写到
scratch，再两两合并，每次只保留前 K 个。

```text
8192 个候选，K=512
    ↓ 4 个 CTA，各自处理 2048 个
四份已排序的 512 项列表
    ↓ 2 个 CTA，两两合并并截取 512 项
两份 512 项列表
    ↓ 1 个 CTA，合并并截取
最终 512 个编号
```

合并不再完整排序。对于左列表中位置 i 的元素，用二分查找统计右列表有多少项
排在它前面；二者相加就是合并后的名次。右列表同理。排序包含编号这个次关键字，
所以有效项有唯一名次，各线程独立写入，不需要 atomic。空位的同值哨兵固定由
左列表优先，保证不足 K 个有效元素时仍完整覆盖输出。

每层由 stream 顺序保证前一层完成。两个 scratch 区交替读写，避免跨 CTA 原地覆盖。
P=`ceil(C/2048)` 时，共一次局部筛选加 `ceil(log2(P))` 次合并 launch；奇数块数也支持。
所有调度只依赖张量形状，位置在 GPU 执行时读取，适合 CUDA Graph。

## 接口和 scratch

| 参数 | 类型、形状 | 含义 |
|---|---|---|
| scores | FP32 `[N,C]` | 只读分数，C 是压缩池容量 |
| start | device I32 `[1]` | 只读绝对位置，不自增 |
| workspace | 可写 I32 `[W]` | 原始 scratch 存储，内容不属于持久模型状态 |
| indices | 可写 I32 `[N,K]` | 排序后的压缩行号和 `-1` 填充 |

先调用 `v4_indexer_topk_workspace_words(N,C,K)`，再分配足够的 I32 存储。
W 的单位是四字节 word，不是 byte：

```text
C ≤ 2048: W = 1                // 接口保留一字，kernel 不使用
C > 2048: W = 4 * N * P * K    // 两个区，每项为 FP32 score + I32 ID
```

| 场景，K=512 | Scratch |
|---|---:|
| 单 token，8192 个候选 | 32 KiB |
| 单 token，32768 个候选 | 128 KiB |
| 单 token，262144 个候选 | 1 MiB |
| 128-token chunk，8224 个候选 | 5 MiB |

示意调用，省略已有 Q/K/score buffer 的准备：

```rust,ignore
let words = Cuda::v4_indexer_topk_workspace_words(n, capacity, 512)?;
let mut workspace = Tensor::<i32, _>::zeros([words], device)?;
let mut indices = Tensor::<i32, _>::zeros([n, 512], device)?;
Cuda::v4_indexer_scores(&scope, &q, &keys, &weights, &start, &mut scores)?;
Cuda::v4_indexer_topk(&scope, &scores, &start, &mut workspace, &mut indices)?;
```

N,C>0，1≤K≤512；K 可以大于 C。张量连续、至少四字节对齐，位于 scope 的设备。
workspace 可以比分配查询结果更大，但必须是一维；所有可写参数都不能与其他参数
重叠。scratch 必须由当前调用独占，不得给并发 stream 复用；张量存活至异步执行完成。
算子内部不分配显存、不下载设备位置、不做 CPU 同步。

负 start、末位置超过 I32 上限、完成块数超过 C 时，indices 全为 `-1`；scratch
内容未指定。形状、容量、alias 和 grid.x 超限由 Rust 返回错误。scores 始终不改写。

## 验证与测量

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test -p infer-backend-cuda --test v4_indexer_topk --test v4_indexer \
  --test v4_csa --test v4_hca --test v4_swa \
  -- --ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo run -p infer-backend-cuda --example v4_indexer_topk_bench
```

独立 CPU 参考对有效候选执行标准排序，直接按 FP32 数值和编号比较，要求 GPU
输出编号逐项相同，没有浮点容差。覆盖所有局部排序尺寸、多个 K、非整块容量、
奇数归并树、26 万候选、跨块同分、正负零、NaN/无穷、因果边界、无效位置与容量、
scratch 长度/stride/alias、输出与 scratch 前后保护区。

打分→选择测试采用可精确计算的 64 头输入，分别验证分数、CPU 排序和输出编号。
完整、分块、逐 token 流程逐位一致；动态 Graph 更新位置与分数，包含请求重启、
有效候选由多变少和填充覆盖，检查显存分配计数不变。

微基准 K=512，64 个索引头，CUDA events 测七组、每组五次 Graph 重放，取中位数。
Prefill 每个 Graph 含 8 次调用，decode 含 32 次。分别计时打分、top-k、两者串联；
不同阶段独立测量，受时钟和调度影响，单项时间不一定恰好相加。
每个场景都将完整结果与 CPU 排序逐项对照。

2026-09-17，RTX 4070 Ti SUPER（16GB，sm_89），CUDA `-O3`：

| 模式 | N | start | C | top-k | 打分 + top-k |
|---|---:|---:|---:|---:|---:|
| Prefill | 128 | 0 | 32 | 7.168 µs | 13.082 µs |
| Prefill | 1024 | 0 | 256 | 43.290 µs | 137.574 µs |
| Chunk prefill | 128 | 32768 | 8224 | 74.342 µs | 611.610 µs |
| Decode | 1 | 127 | 32 | 4.634 µs | 8.934 µs |
| Decode | 1 | 1023 | 256 | 4.678 µs | 9.483 µs |
| Decode | 1 | 32767 | 8192 | 13.971 µs | 19.814 µs |
| Decode | 1 | 131071 | 32768 | 22.545 µs | 41.101 µs |
| Decode | 1 | 1048575 | 262144 | 39.057 µs | 220.877 µs |

Decode 测试中 C 等于可见候选数，不是在最大预留容量下测所有历史长度。
短历史可能让 top-k 只有少量有效输出，但本表仍分配固定宽度 K=512 并写入填充，
与固定形状 Graph 的用法一致。

6 项 top-k 测试及原有 23 项 Indexer/SWA/HCA/CSA 回归测试全部通过。
Rustfmt、diff 空白检查和 Clippy 通过；Clippy 沿用对已有 `dead_code`、
`extra_unused_type_parameters`、`manual_is_multiple_of` 告警的排除。

ptxas：局部排序的 256/512/1024/2048 槽位直接输出路径，分别使用
37/39/48/69 个寄存器；2048 槽位写候选路径为 71 个寄存器。
这些路径 shared memory 均为 9,280 bytes；合并路径为 32 个寄存器、8,192 bytes。
所有 kernel 无 spill、无 stack frame。本轮未运行 Compute Sanitizer；
此前环境的 WSL/WDDM 调试接口初始化失败。

这些计时不包含输入传输、投影、索引压缩、Hadamard/量化或主注意力，不能当作
完整 Indexer 或完整模型吞吐。原始结果在 `target/v4-indexer-topk-bench.jsonl`。

下一步是消费这些编号的稀疏联合注意力；打分与局部选择的融合可在后续测量后推进，
以减少 `[N,C]` 分数矩阵和分块候选的显存往返。

参考：[DeepSeek Indexer](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)、
[NVIDIA CUB BlockRadixSort 的稳定性定义](https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_radix_sort.cuh)。
