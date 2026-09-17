# DeepSeek V4：16GB 单卡最小验证

这是 RustInfer 的独立架构验证入口 `rustinfer-v4-tiny`。用本地生成的随机权重，
在一张卡上核对 V4 的 prefill、分块 prefill 和逐 token decode。
不下载真实模型，不需要训练。随机输出没有语言能力。

| 预设 | 参数量 | 权重文件数据 | 用途 |
|---|---:|---:|---|
| `micro` | 80,581 | FP32 325,412 bytes，含 hash 表 | 无 GPU 的固定回归样本 |
| `tiny`（默认） | 859,245 | BF16 1,730,786 bytes，含 hash 表 | 单卡开发与 BF16 对齐 |

默认 tiny：4 层、hidden=128、词表=256、4 个路由专家＋1 个共享专家、top-k=2、
专家中间维度=64、4 个注意力头、head_dim=32、mHC=4。
注意力序列是 SWA/CSA/HCA/CSA；前三层 hash 路由，最后一层普通路由。
保留 CSA 压缩率 4 和 HCA 压缩率 128，滑窗缩为 16，最大上下文 256。

## 执行范围

模型组合代码在 `crates/infer-worker/src/models/deepseek_v4/`。
矩阵乘法调用现有 CPU/CUDA 后端；mHC、路由选择、压缩池、注意力的标量计算和缓存
暂在主机端执行，矩阵乘法完成后显式同步。这条路径便于查看中间结果，会有设备往返开销。
它还没有接入 `DecoderModel`、服务调度器或 CUDA Graph，不能用于推断生产吞吐。

参考目标固定为 **Transformers 5.12.0 的 eager、非量化 V4**：

- shared-KV、partial RoPE 与输出逆旋转、attention sink、分组低秩输出投影。
- CSA 的重叠压缩、Lightning Indexer 和 HCA 的压缩池。
- mHC Sinkhorn 混合与最终 residual-stream 合并。
- hash 路由、sqrt(softplus) 路由、只影响选专家的 correction bias、共享专家和 clamped SwiGLU。
- 每个请求独立的滑窗、未完成压缩块和上一压缩块状态。

本阶段只接受脚本生成并标记 `rustinfer_tiny=true` 的受限配置。
完整模型、量化权重/KV、YaRN 长上下文、MTP、多卡、投影 bias 和 dropout 均不在范围内。
因此这里的通过结果**不代表官方 FP4/FP8 权重已经适配完成**。

## 本地生成与运行

在仓库根目录准备 Python 依赖（模型创建和运行过程不访问 Hub）：

```bash
uv venv .venv
uv pip install --python .venv/bin/python \
  'torch==2.11.0' 'transformers==5.12.0' 'safetensors==0.8.0' 'numpy>=2,<3'

.venv/bin/python scripts/deepseek_v4_tiny.py \
  --output target/deepseek-v4-tiny --device cuda --dtype bfloat16
```

也可通过 `pyproject.toml` 的 `deepseek-v4-reference` extra 准备依赖。
生成脚本要求输出目录为空，避免覆盖已有样本。
目录包含 `config.json`、`model.safetensors`、`reference.safetensors` 和 `manifest.json`；
manifest 记录 seed、依赖版本、参数量、权重字节数、固定输入、分块方式及 PyTorch 显存统计。

使用仓库现有 CUDA 构建环境，4070 Ti SUPER 对应 `sm_89`：

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_89 \
  cargo build -p infer-worker --bin rustinfer-v4-tiny

LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  target/debug/rustinfer-v4-tiny \
  --model target/deepseek-v4-tiny --backend cuda \
  > target/deepseek-v4-report.json
```

`--backend cuda` 在 GPU 上执行 GEMM，其余参考运算在主机端执行。
该入口把 CUDA workspace 和内存池保留上限各设为 8MiB，关闭 graph arena。
参考生成和 Rust 运行可以顺序执行，避免同时保留两份运行时。

可选 `--dump target/deepseek-v4-dump` 保存 Rust 的每层输出和 logits，便于定位差异。
输出 JSON 包含每种执行方式、每层误差、最差元素、阈值和最终压缩池条目数；
任一指标超限会返回非零退出码。请同时检查退出码与 `passed`。

## 无 GPU 回归

仓库附带 `micro` 的 FP32 权重与独立 Transformers 参考结果，测试无需安装 Python：

```bash
cargo test -p infer-worker --no-default-features --test deepseek_v4

cargo run -p infer-worker --no-default-features --bin rustinfer-v4-tiny -- \
  --model crates/infer-worker/tests/fixtures/deepseek_v4 --backend cpu
```

重新生成 micro 到新目录：

```bash
.venv/bin/python scripts/deepseek_v4_tiny.py \
  --output target/deepseek-v4-micro --preset micro --dtype float32 --device cpu
```

若使用 CUDA 验证 FP32，启动进程前设置 `NVIDIA_TF32_OVERRIDE=0`：
现有 CUDA FP32 GEMM 默认启用 TF32，会超过严格 FP32 对齐阈值。

## 验收与限制

固定输入长 137，分块为：

- 整段：`[137]`。
- 分块：`[3, 1, 11, 1, 111, 1, 1, 8]`。
- decode：`[127]` 后接 10 次单 token。

这些输入覆盖 CSA 首块与重叠状态、滑窗淘汰、HCA 在 128-token 边界首次生成条目，
以及 prefill/decode 切换。每次比较全部层的 mHC 流和全部位置的 logits。
另有独立缓存、非法 token、上下文越界和不支持配置的回归测试。

FP32 门槛为最大绝对误差 `2e-5` 且相对 L2 `2e-4`；BF16 分别为 `0.025` 和 `0.03`。
随机样本的索引分数避开 ReLU 零值并列区，路由 correction bias 留出明确间隔，
避免不同舍入或 top-k 并列处理改变离散选择。它验证这些明确选择下的计算路径；
真实权重下接近并列的路由行为、量化误差与长上下文仍需独立验证。

2026-09-16 在 RTX 4070 Ti SUPER 16GB 的本地验证：

- FP32 micro 的 CPU 与禁用 TF32 的 CUDA 路径通过三种执行方式。
- BF16 tiny 的三种执行方式通过，logits 最大绝对误差 `0.012207`，相对 L2 约 `0.00733`。
- PyTorch tiny 参考的峰值 allocated 约 11.23MiB、reserved 24MiB；这不包含 CUDA 上下文。
- Rust tiny 运行时，整卡显存采样从 2211MiB 升到 2448MiB，增量约 237MiB。
  这是整卡采样估计，包含桌面和其他常驻占用，不是精确的进程峰值。

架构与参考代码：
[官方模型](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、
[Transformers V4 实现](https://github.com/huggingface/transformers/tree/v5.12.0/src/transformers/models/deepseek_v4)。
