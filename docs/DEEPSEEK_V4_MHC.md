# V4 mHC：四条残差流的 CUDA 混合

`FusedOps::v4_mhc_pre`、`v4_mhc_post`、`v4_mhc_head` 已提供独立 CUDA 实现，
覆盖 prefill 和 decode。BF16 激活，FP32 映射权重、门控系数和 Sinkhorn 状态；
固定 `hc_mult=4`，支持偶数 `2<=hidden_size<=8192`，包括 tiny 的 128 和 Flash 的 4096。
这里的 mHC 是残差流混合；HCA 是另一种压缩注意力。

当前仍是算子接口。tiny 的 `hyper.rs` 保留主机参考路径，完整 V4 执行器尚未接入。
本实现不包含 Attention/FFN、最终 RMSNorm 或 lm_head，也不代表真实量化模型已可运行。

## 数学与精度边界

每个 token 的输入 `X[4,D]` 展平为 `x[4D]`：

```text
inverse_rms = 1 / sqrt(sum(x*x)/(4D) + norm_eps)
mix = (W @ x) * inverse_rms                 W: [24,4D], FP32
pre  = sigmoid(mix[0:4] * scale[0] + base[0:4]) + hc_eps
post = 2 * sigmoid(mix[4:8] * scale[1] + base[4:8])
C = row_softmax(reshape(mix[8:24] * scale[2] + base[8:24], [4,4])) + hc_eps
C = column_normalize(C, denominator_eps=hc_eps)
repeat iters-1 times: row_normalize(C), column_normalize(C)
collapsed[d] = sum_i pre[i] * X[i,d]
```

`collapsed` 是 Attention/FFN 的输入；输出 `branch[D]` 再由 Post 放回四条流：

```text
output[j,d] = sum_i C[i,j] * X[i,d] + post[j] * branch[d]
```

注意 `C[i,j]` 的方向：这是 `C^T @ X`。Head 仅计算四个 pre 系数并合并四条流，
映射权重为 `[4,4D]`，不执行 Sinkhorn。

数值目标沿用仓库参考版和 **Transformers 5.12.0** 的 BF16 eager 路径。
Pre 的 RMS 标量移到点积之后，数学等价，但 FP32 累加/舍入次序有差异；因此不承诺
任意输入均逐位一致。FP32 权重不会被截断为 BF16/TF32。
Post 保留参考版的三个舍入边界：先将 post/C 转为 BF16，残差矩阵乘结果和 branch
乘积分别舍入为 BF16，再相加并舍入为 BF16。不能将整段直接融合为一次 FP32 FMA。

[DeepSeek 官方脚本](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)
的 Post 使用 FP32 系数及中间值、最后才转回激活类型，与上述 Transformers BF16
路径存在精度边界差异；本接口明确选择后者，与已有 tiny oracle 保持一致。
[Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
使用四条流、hidden=4096、20 次 Sinkhorn；接口也支持 1–20 次，用于小模型验证。

## GPU 调度

Pre 和 Head 各两次 launch；Post 一次 launch。

1. **投影与 RMS 统计**：沿 `4D` 按 256 个元素切块，避免 decode 只有一个 CTA。
   每个 CTA 用 256 线程处理最多四个 token，复用同一批 FP32 权重。
   Warp 分工计算映射点积，同时统计平方和，输出 FP32 部分和。
2. **归约、Sinkhorn、collapse**：每个 token 一个 128 线程 CTA。
   部分和归约后，warp 中的 16 个 lane 分别持有 `4×4` 矩阵的一个元素。
   XOR shuffle 1/2 完成行归约，4/8 完成列归约；20 轮迭代留在寄存器中。
   其余线程参与 D 维的加权合并。Head 使用同一套模板，编译时移除 Sinkhorn。
3. **Post**：沿 D 每 256 列一个 CTA；每个线程读四条输入流一次，计算四条输出流。
   显式转换和非融合加/乘保证 BF16 舍入边界。

没有原子累加、内部显存分配、CPU 数据回读或同步。kernel 选择和 grid 只依赖形状，
支持 CUDA Graph；连续的 Pre→Post→Head 可复用同一 scratch，依靠同一 stream 的顺序。
映射目前采用 SIMT FP32，针对 24/4 个小输出维度复用权重。现有通用 FP32 matmul
默认允许 TF32，直接替换会改变这里的精度边界；本轮未宣称此实现优于严格 FP32 cuBLAS。

## 接口与内存

| 参数 | Pre | Head |
|---|---|---|
| residual | BF16 `[N,4,D]` | 相同 |
| weight | FP32 `[24,4D]` | FP32 `[4,4D]` |
| scale / base | FP32 `[3]` / `[24]` | FP32 `[1]` / `[4]` |
| collapsed | BF16 `[N,D]` | 相同 |
| post / comb | FP32 `[N,4]` / `[N,4,4]` | 无 |
| workspace | FP32 `[W]` | FP32 `[W]` |

Post 读取 residual、branch `[N,D]`、post、comb，写 BF16 `[N,4,D]`，不需要 scratch。

通过 `v4_mhc_workspace_floats(N,D,head)` 查询元素数，单位是 FP32 元素：

```text
P = ceil(4D/256)
Pre:  W = N*P*25
Head: W = N*P*5
```

D=4096 时，单 token Pre scratch 为 6400 bytes，Head 为 1280 bytes；
1024-token Pre 为 6.25 MiB。所有 scratch 均由调用方提前分配，内容不属于持久状态。

要求 N>0、`N*P<=i32::MAX`，eps 有限且正数。输入须为有限值，中间值在 FP32
范围内；不额外扫描设备数据。张量必须连续、四字节对齐、位于 scope 设备。
所有可写张量与其他参数的范围均不得重叠，不支持原地更新。调用方负责 stream
顺序和缓冲区生命周期；并发调用使用独立 scratch 和输出。参数检查复用 `v4_common`。

## 验证与复现

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  V4_TEST_PROFILE=dev bash scripts/v4_regression.sh

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo test -p infer-backend-cuda --test v4_mhc -- --include-ignored --test-threads=1

CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  cargo run -p infer-backend-cuda --example v4_mhc_bench
```

6 个 GPU 测试加 1 个 workspace 单元测试，覆盖：

- 独立 FP64 标量 oracle：D=2/6/32/128/320/4096/8192，非整 token tile，迭代 1/2/20。
- 非对称矩阵与有方向的流置换，Post 的 BF16 舍入逐位对齐。
- 零输入、饱和正负 logits，Sinkhorn 有限性和受 epsilon 影响的归一化。
- 137-token 整段/分块/decode 输出逐位一致，无内存池增长。
- 捕获 Pre→Post→Head，重放前更新输入、权重、scale/base 和 branch。
- scratch/output 前后哨兵、形状、连续性、对齐、别名、epsilon、迭代次数和容量拒绝。

独立 Transformers 对照使用已有 tiny 开发环境，无模型下载：

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_89 -shared -Xcompiler=-fPIC \
  crates/infer-backend-cuda/src/kernels/v4_mhc/v4_mhc.cu \
  -o target/libv4_mhc_reference.so
.venv/bin/python scripts/check_v4_mhc_reference.py --library target/libv4_mhc_reference.so
```

2026-09-18，RTX 4070 Ti SUPER 16GB / sm_89，torch 2.11.0+cu130、Transformers 5.12.0：
N/D=1/128、137/128、5/4096、1/8192 全部通过。Pre 的 BF16 输出在这些样本上完全一致；
post 系数最大绝对误差 `3.58e-7`，comb `2.98e-7`；以相同系数输入的 Post 逐位一致；
Head 最大绝对误差 `3.82e-6`。

sm_89 ptxas：投影 Pre/Head 分别 48/40 registers；finish 34/38 registers，
116/36 bytes shared；Post 30 registers。所有 kernel 均无 stack/spill。
尚未完成跨 GPU 架构或 Compute Sanitizer 验证。

benchmark 对各阶段和合成 Pre→Post→Head 分别捕获图，以 CUDA events 记录
7 组、每组 5 次 replay 的中位数；decode 图包含 32 次调用，较长 prefill 图包含 8 次。
另比较相同 kernel 逐 token 执行的 Pre，并校验两者输出逐位相同。
这些计时包含 FP32 投影，不包含传输、分配或真正的 Attention/FFN。
首轮显卡在程序退出后仍有约 86–88% 利用率和 10GB 占用，计时明显受其他负载
干扰，已弃用。负载下降到桌面状态后复测两次，阶段延迟如下（单位 µs）：

| N / D | Pre | Post | Head | 合成 Pre→Post→Head |
|---|---:|---:|---:|---:|
| 1 / 128 | 6.970–6.989 | 1.024–1.043 | 3.249–3.275 | 11.168–11.315 |
| 1 / 4096 | 12.063–12.114 | 1.165–1.169 | 8.056–8.070 | 19.738–21.329 |
| 128 / 4096 | 56.115–56.237 | 6.784–6.861 | 20.986–21.018 | 83.711–84.889 |
| 1024 / 4096 | 404.141–416.450 | 130.048–130.918 | 124.054–128.845 | 928.536–936.155 |

这是非独占桌面 GPU 上的两次观测范围，不是统计置信区间；单独阶段与流水线的缓存
行为不同，不能直接相加。逐 token Pre 只用于检查批处理收益，不作为纯 GEMM 或
其他框架的比较基线。[完整两轮输出](benchmarks/v4_mhc_sm89.md)保留 N=4 和 tiny prefill
等其余形状、scratch 大小及逐 token 对照结果。
