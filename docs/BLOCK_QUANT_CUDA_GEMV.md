# CuTe DSL 块量化 GEMV

已将单 token 的量化 Linear 接入 CUDA。后续[多 token GEMM](BLOCK_QUANT_CUDA_GEMM.md) 也已接通；本页保留 GEMV 阶段记录：

```text
Linear::forward(BlockQuantProjection)
  → 按输出通道切分不同格式的权重、bias、输出
  → Cuda::matmul_block_quant
  → CuTe DSL GEMV：压缩块解码 → FP32 乘加、warp 归约 → bias → 输出类型转换
```

## 范围与调用约定

- 输入 `[1,K]`，权重逻辑形状 `[N,K]`，可选 bias `[N]`，输出 `[1,N]`。
- 13 种格式：Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS。
- 激活、bias 和输出使用同一种类型：FP32、FP16 或 BF16，共 39 个 AOT 内核。
- 支持权重行切片、奇数地址的压缩权重、输入／bias／输出的非零 offset 和非连续 stride；输入和 bias 允许广播 stride。
- 拒绝输出元素重叠、输出与任意输入共享 Storage、错误形状，以及跨 CUDA 设备／配置的张量。
- `M=0` 为空操作；`M>1` 现自动分发到后续接入的 GEMM。块量化组件仍仅支持 TP1。
- 未启用 `cute-dsl` 或实际 GPU 与 AOT 架构不符时返回 `Unsupported`，不回退 CPU。

## 实现

一 CTA 使用 128 线程，四个 warp 分别计算四个输出通道。每个 lane 以 32 为步长遍历 K，直接将量化值解码到寄存器并累加；warp 内归约后，仅 lane 0 写出。尾部不足四行时以 warp 为单位屏蔽。

复用 Embedding 的 CuTe 解码函数和每个 CUDA 配置中的 17,424 字节 IQ 码表，不维护另一套格式逻辑。内核使用 64 位地址计算，不假定压缩块具备 FP16 对齐。无需整张浮点权重、临时显存、原子操作或共享内存。

FP32 解码、乘法、累加、归约和 bias 相加完成后才转换输出类型；禁用 ptxas FMA 合并。归约顺序与 CPU 串行累加不同，因此 GEMV 使用误差范围比较，不承诺逐位相同。NaN、Inf 按浮点运算传播。

所有模块在 CUDA 配置创建时加载，执行期间只有主机元数据校验和异步 kernel launch，不分配内存、不复制至主机、不同步流，可用于 CUDA Graph 捕获和重放。调用方须遵守异步算子的通用存储生命周期约定。现有 Embedding 的 ID 检查仍有独立的捕获限制。

AOT 编译器校验唯一入口、12 个 64 位参数（5 个指针、7 个整数）、128 线程、目标架构、无共享内存及 ELF cubin。构建仍固定使用 `nvidia-cutlass-dsl==4.7.1`；运行时无 Python 或 llama.cpp 依赖。

这是正确性优先的基础实现，尚未按格式优化块元数据加载、指令重复、向量化或长 K 的并行分工。没有把算子耗时换算为完整模型生成速度。

## 验证命令

```sh
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries

cargo +stable test -p infer-backend-cuda --features cute-dsl \
  --test cute_block_gemv -- --ignored --test-threads=1

RUSTINFER_GGUF_MODEL=/root/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  cargo +stable test -p infer-worker --features cute-dsl --test block_quant_weights \
  block_quant_local_cuda_gemv -- --ignored --nocapture --test-threads=1
```

固定向量测试复用 832 个独立打包的量化块，并按每行 1、2、8、64 块重新组合；覆盖全部格式和 dtype、有／无 bias、切片、非连续布局、非对齐权重、输出哨兵、错误输入、设备配置隔离、广播、空输入、非有限值，以及 Graph 捕获和重复重放。对照 CPU 原生 GEMV，同时以 CPU 解码加 FP64 点积校验误差；界限包含 `sum(abs(x*w))` 缩放的累加误差和目标类型舍入误差。

真实文件测试验证全部 13 种格式的真实行长度及三种 dtype，然后上传第 3 层完整 Q/K/V 权重，使用实际混合格式 `Linear::forward`，对全部输出与 CPU Linear 和 FP64 点积比较。该测试还验证分段 bias 和带 stride 的输出，并记录 CUDA event 计时。

## 本次实测（2026-09-23）

RTX 4070 Ti Super（16 GiB，sm_89）上通过：

- 5 个 GEMV GPU 测试，含 FP16/BF16 在 bias 相加前不能提前舍入的边界案例。
- 真实 GGUF 的全部 13 种格式 × 3 种 dtype。
- 第 3 层完整 IQ4_NL／Q4_K／Q5_K 混合 QKV：逻辑形状 `[14336,5120]`，压缩权重 **41,943,040 字节（40 MiB）**，全部输出验证通过。
- 混合 QKV 相对 FP64 点积最大绝对误差约 **3.3e-7**，相对 CPU FP32 串行实现约 **4.77e-6**。
- 预热后 30 次混合 QKV 的 CUDA event 平均约 **1.478 ms**；包含三次内核发射之间的间隔，使用带 stride 的 FP32 输出。测试采用 Cargo debug 主机代码；CuTe cubin 由 ptxas 编译。这是该算例的基线，不是端到端 tokens/s 或跨引擎性能比较。
- 5 个既有 GPU Embedding 测试、9 个 CuTe RMSNorm 测试和 10 个 worker 块量化集成回归通过。

本次未取得 Compute Sanitizer 内存检查结果；此前该 WSL/WDDM 环境缺少调试接口，详情见 Embedding 文档。完整 GGUF 模型生成和视觉模型装配仍属于后续阶段。

定向 rustfmt 和 `git diff --check` 通过。CUDA 库及 GEMV 测试的 Clippy 检查完成；仅保留已有 RMSNorm／attention／scalar 代码的 9 条告警，新增 GEMV 代码和测试无告警。
