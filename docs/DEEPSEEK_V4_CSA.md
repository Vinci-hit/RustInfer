# V4 CSA 第一步：重叠压缩器 GPU kernel

接在 [HCA](DEEPSEEK_V4_HCA.md) 后，先实现主注意力分支的 CSA compressor。
入口为 `FusedOps::v4_csa_compress`，CUDA 代码在
`crates/infer-backend-cuda/src/kernels/v4_csa/`。单请求、压缩比 4、D=512、
尾部 interleaved RoPE 宽度 64，支持完整 prefill、任意长度分块与单 token decode。
可在单张 16GB 显卡上独立验证，不需要下载完整模型。

这是 CSA 的压缩阶段。完整 CSA 还需要 Lightning Indexer/top-k，以及将选中的
压缩 KV 和最近 128 个原始 KV 放进同一个 softmax 的稀疏注意力。
Indexer 有独立的压缩投影、维度及 Hadamard/FP4 路径，当前入口对应主注意力的
512 维压缩池，不能直接充当完整 Indexer。

## 为什么不能只把 HCA 的 128 改成 4

HCA 每个 token 的 values/gates 投影宽度为 D，块之间不重叠。
CSA 每个 token 投影出 2D，分成前半 A 与后半 B：

```text
tokens 0..3:    A0  B0
tokens 4..7:    A1  B1
tokens 8..11:   A2  B2

压缩 KV[0] = pool(缺失历史, B0)   位置 3 完成，RoPE 位置 0
压缩 KV[1] = pool(A0, B1)         位置 7 完成，RoPE 位置 4
压缩 KV[2] = pool(A1, B2)         位置 11 完成，RoPE 位置 8
```

每个 A/B 都有四行。首块只有四个有效输入，其余块有八个。
重叠发生在源 token 上；前后两个压缩块使用的是该 token 的不同投影半部。
不能把它实现成普通的 8-token 滑窗均值，也不能将两个半部各自 softmax 后取平均。

通道 d 的公式：

```text
score[t,d] = gates[t,half*D+d] + ape[t%4,half*D+d]
pooled[d] = sum_t exp(score[t,d]-m[d]) * values[t,half*D+d]
            / sum_t exp(score[t,d]-m[d])
compressed = partial_RoPE(RMSNorm(pooled), position=block_id*4)
```

其中前块取 half=0，当前块取 half=1。APE 是每半部各自学习的四行偏置。
Softmax 沿 token 轴进行，各通道拥有自己的权重和分母。

## 18 KiB 状态与融合

每个池保存 FP32 最大值 m、相对最大值的权重和 z、加权值和 u。
增量更新沿用 HCA 的 online softmax。合并前块 P 和当前块 Q 时：

```text
m = max(mP, mQ)
a = exp(mP-m), b = exp(mQ-m)
z = a*zP + b*zQ
u = a*uP + b*uQ
pooled = u/z
```

实际只需计算一个指数，另一个缩放因子为 1。首块的空历史直接跳过。
必须保留的三组统计量如下：

| state 第一维 | 内容 | 用途 |
|---|---|---|
| 0 | 前一个完整块的 A | 当前块输出所需的重叠历史 |
| 1 | 当前块已到达 token 的 A | 完成后成为下一块的历史 |
| 2 | 当前块已到达 token 的 B | 与组 0 合并，产生当前输出 |

每组的第二维依次是 m、z、u，总共 `3*3*512*4 = 18432 bytes`。
官方教学路径的两份 `[8,1024]` FP32 values/scores 状态共 64 KiB；这里用
18 KiB 统计量替代。此比较仅指压缩状态，不包含投影输入、权重和压缩 KV 池。

- Decode：256 线程、每线程两个通道，一个 kernel 更新两组当前统计量。
  每第四步合并历史，融合 RMSNorm、RoPE 和 BF16 输出，然后将 A 转存为历史。
- Prefill：每个完成块一个 CTA，并行读取前一块 A 与当前块 B，生成压缩行。
  第二个顺序 kernel 提交最终状态，最多重算八个 token 的统计量。
  输出 kernel 只读旧状态，避免首块读取前缀时被尾部提交覆盖。
- 不创建临时 GPU buffer，不读回位置，不同步 CPU；位置在设备执行时读取，
  支持 CUDA Graph 更新输入与位置后重放。

## 接口与状态契约

| 参数 | 类型、形状 | 含义 |
|---|---|---|
| values/gates | FP32 `[N,1024]` | 两个投影 GEMM 的结果，N>0 |
| ape | FP32 `[4,1024]` | learned position bias |
| norm | FP32 `[512]` | RMSNorm 权重；eps 单独传入，有限且 >0 |
| rope | FP32 `[C,32,2]` | cos/sin，行 b 编码块起点 b*4；调用者准备 YaRN 参数 |
| start | device I32 `[1]` | chunk 的绝对起点或 decode 位置，不在此自增 |
| state | 可写 FP32 `[3,3,512]` | 三组 m/z/u |
| compressed | 可写 BF16 `[C,512]` | 绝对块编号索引，C>0 |

输出链是 FP32 pooling → BF16 → FP32 RMSNorm → BF16 → FP32 RoPE → BF16。
当前保留未量化 BF16 参考的舍入边界，尚不包含官方非 RoPE 部分的 FP8/QAT，
也不包含投影 GEMM、Indexer 或后续注意力。压缩器本身主要是归约与逐元素运算，
没有为 Tensor Core 构造额外矩阵；投影 GEMM 是独立的矩阵乘法阶段。

从 start=0 开始时，旧 state 可未初始化。非零块边界仍然需要前块 A 的统计量；
这与 HCA 的独立块不同。满块后组 0 更新、组 1/2 归为空池 `(-inf,0,0)`。
从块中间继续还需要正确的当前前缀。请求须独占状态和压缩池，按同一 stream 调用。
仅完成的块写入压缩池，历史行和未来行保持原样；未来行可以填 NaN 验证未被提前读取。

`C` 是完成块容量，因此 C=2 可以接受前 11 个 token，第 12 个才超出容量。
负 start、末位置超过 I32 上限、完成块数超过 C 时，整个调用不写 state 或压缩池。
形状、连续性、设备、四字节对齐、可写参数 alias 和 eps 由 Rust 层检查。
输入投影、权重及有效历史须有限，张量须存活至异步执行完成。

## 验证与微基准

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test -p infer-backend-cuda --test v4_csa --test v4_hca --test v4_swa \
  -- --ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo run -p infer-backend-cuda --example v4_csa_bench
```

测试使用独立 FP64 两遍 softmax，直接拼接八个源 token 作为参考，不复用 online
累加或合并实现。参考在 pooling、norm 和 RoPE 后模拟 BF16 舍入；RoPE 的误差界
按两个旋转项的幅度传播，避免相减接近零时误用最终结果的相对误差。
另有独立 FP64 状态参考，检查三组 m/z/u 的值与空池归位。

覆盖首块缺失历史、不同通道的极端 gates、learned APE、不同半部的联合分母、
块起点 RoPE、只影响下一个块的 A 半投影、未来输入因果性；枚举 144 种前缀与
chunk 长度组合，验证边界状态。Graph 测试更新输入和位置，并检查重启请求及
分配器计数。完整、分块和逐 token 路径要求输出与最终状态逐位一致。

微基准通过 CUDA events 测七组、每组五次 Graph 重放，取中位数；短 prefill
每张 Graph 组成 32 次调用以减少主机提交间隙。逐 token 基线也使用 Graph。
Decode 使用固定可重复的四步循环，包含每四步一次的重叠输出，而非持续计时
反复追加同一中间 token 的无效状态。单独报告块起点无输出调用的耗时。
所有计时不含投影 GEMM、CPU/GPU 传输、Indexer、attention 或模型其他部分。

2026-09-17，RTX 4070 Ti SUPER（16GB，sm_89），CUDA `-O3`：

| token 数 | 并行压缩 prefill | 逐 token 压缩 Graph | 加速比 |
|---|---:|---:|---:|
| 128 | 5.958 µs | 237.773 µs | 39.91× |
| 512 | 8.045 µs | 935.731 µs | 116.32× |
| 1024 | 10.643 µs | 1841.152 µs | 172.99× |
| 4096 | 18.611 µs | 6572.973 µs | 353.17× |

这里比较的是两个本地 kernel 路径。加速主要来自并行处理各块，并将 N 次单步
launch 减少为两次；不代表相对于官方优化实现的加速，也不代表整个模型的加速。
所有规模的输出与最终状态逐位一致。原始结果在 `target/v4-csa-bench.jsonl`。
Decode 块起点无输出调用为 1.227 µs；包含四步一次输出的完整循环平均为
1.376 µs/token。CUDA events 不计主机侧 API 开销，实际服务吞吐需集成后验证。

6 项 CSA、5 项 HCA、6 项 SWA GPU 测试通过。Rustfmt、diff 空白检查和 Clippy
通过；Clippy 沿用对已有 `dead_code`、`extra_unused_type_parameters`、
`manual_is_multiple_of` 告警的排除。
ptxas 报告 decode/完成块各 40 个寄存器、2,096 bytes shared memory，
尾部状态提交 43 个寄存器、无 shared memory，均无 spill、无 stack frame。
本轮未运行 Compute Sanitizer；之前环境的 WSL/WDDM 调试接口初始化失败。

参考：[DeepSeek 官方 Compressor](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)。
