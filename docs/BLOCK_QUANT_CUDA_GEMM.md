# CuTe DSL 块量化 GEMM

`Cuda::matmul_block_quant` 现在支持 `[M,K] × [N,K]ᵀ → [M,N]`。
`M=1` 保留现有 GEMV；`M>1` 使用 GEMM；`M=0` 校验后为空操作。
`Linear::forward(BlockQuantProjection)` 自动沿输出通道切分混合格式的权重、bias 和输出，不需要模型层另外调度多 token 算子。

## 内核

这是融合解量化的 **SIMT GEMM**。每个 CTA 有 128 线程，覆盖 **8 个 token × 4 个输出通道**。四个 warp 各负责一个输出通道，每个 lane 以 32 为步长遍历 K：量化权重只解码一次，供最多八个 token 的 FP32 累加器复用。完成 warp 归约及 bias 相加后，才转为输出 dtype。

支持 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K、Q8_0、IQ2_XXS/IQ2_XS/IQ2_S、IQ3_XXS/IQ3_S、IQ4_NL/IQ4_XS，输入／bias／输出类型支持 FP32、FP16、BF16，共 39 个 GEMM AOT 内核。沿用已有解码函数和每配置一份的 IQ 码表。没有 llama.cpp 依赖。

不会展开整张浮点权重，也不分配临时显存；无 shared memory、原子累加和主机同步。每个混合格式分段发射一个 GEMM 内核，而非为每个 token 分别发射 GEMV。使用 64 位地址，M、N 的尾块均有边界保护，K 仅需符合量化格式块大小要求。

该版本尚未使用 Tensor Core、共享内存分块或流水线。禁用 FMA 合并，保留现有 FP32 解码／累加语义；它是功能和性能基线，不代表完整推理框架或长提示词的最终性能。

## 接口和存储约定

- 权重保持 `BlockQuantWeight` 压缩存储，可取行切片、共享底层 Storage，允许奇数地址的压缩块。
- 输入、bias 和输出支持非零 offset 与 stride；矩阵可以转置或带 padding。输入和 bias 支持广播 stride。
- 输出元素必须互不重叠，且不能与输入／bias／权重共享 Storage。
- 混合投影在任何分段写入前检查**完整输出**的重叠情况，防止分段各自合法、但跨 token／分段互相覆盖。判定函数统一放在 `infer_core::types::matrix_elements_are_disjoint`，Embedding 和 CUDA matmul 共用。
- 验证实际 CUDA 设备和配置、形状、dtype 大小、整数范围及合并后的网格大小；未开启 `cute-dsl` 或 AOT 架构不匹配时返回 `Unsupported`。
- 模块初始化时加载，执行期间只校验主机元数据并发射内核，支持 CUDA Graph 首次捕获和更改输入后的重放。异步存储生命周期约定与现有 CUDA 算子一致。块量化组件仍为 TP1。

GEMV/GEMM 的公共 Rust 校验与分发收拢至 `src/block_matmul.rs`；两个 AOT 清单共用规格结构。编译器校验 GEMM 的 15 个 64 位参数（5 个指针、10 个整数）、128 线程、单入口、目标架构及无共享内存。

## 验证

```sh
export PATH="$PWD/.venv/bin:$PATH"
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries

cargo +stable test -p infer-backend-cuda --features cute-dsl \
  --test cute_block_gemm --test cute_block_gemv -- --ignored --test-threads=1

RUSTINFER_GGUF_MODEL=/root/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  cargo +stable test -p infer-worker --features cute-dsl --test block_quant_weights \
  block_quant_local_cuda_gemm -- --ignored --nocapture --test-threads=1
```

独立打包向量覆盖全部 13 种格式和三种 dtype，M=2/7/8/9/17/33，包含 M/N 尾块、每行 1/2/3/8/16/64 个量化块、权重行切片、非对齐地址、转置／padding／广播、有无 bias、输出哨兵。结果对照原生 CPU matmul 与 CPU 解码后的 FP64 点积。另验证非法输入、设备配置隔离、首次 Graph 捕获、更改输入后的重放、FP16/BF16 bias 舍入边界，以及 NaN/Inf 不污染其他 token。

布局判定用小矩阵地址枚举进行穷举核对，另有 CPU 混合投影测试验证跨分段重叠必须在写出前失败。

真实 GGUF 测试先验证全部 13 种格式的真实 K、三种 dtype、M=9；再将第 3 层完整 Q/K/V（IQ4_NL/Q4_K/Q5_K，40 MiB 压缩权重）组装成 Linear，通过 Graph 计算 17 个 token。全部输出与已验证的逐 token GEMV 比较，另对每个投影首尾、tile 边界及中间位置的 15 个输出通道、全部 17 个 token（共 255 个结果）使用 CPU Linear 与 FP64 独立校验。性能测试使用相同完整投影，对 M=2/8/17/64 比较 GEMM 与逐 token GEMV；CUDA event 包含主机发射间隔，不等于端到端 tokens/s。

## 本次结果（2026-09-23）

RTX 4070 Ti Super 16 GiB（sm_89）上：5 个 GEMM GPU 测试、5 个 GEMV 测试、5 个 Embedding 测试、9 个 CuTe RMSNorm 测试、44 个 core 测试、11 个 worker 块量化测试及 1 个真实 GGUF GEMM 测试，共 **80 项通过**。

真实 17-token QKV 的全部 **243,712 个输出**与 GEMV 逐元素一致。255 个 CPU／FP64 独立抽样结果的最大绝对误差分别约 **1.97e-6／2.0e-7**；13 种真实格式 × 三种 dtype 的多 token 测试也通过。

GPU 回归结束后单独重测的 CUDA event 均值如下；每项预热三次，计时十次。相同模型权重、FP32 激活和连续输出，Cargo debug 主机代码，ptxas 编译的 CuTe cubin。环境为 WSL/WDDM，结果会受时钟、系统负载和发射间隔影响：

| token 数 M | GEMM / 批 | 逐 token GEMV / 批 | 本次加速比 |
|---:|---:|---:|---:|
| 2 | 0.701 ms | 0.835 ms | 1.19× |
| 8 | 0.740 ms | 3.136 ms | 4.23× |
| 17 | 2.141 ms | 6.461 ms | 3.02× |
| 64 | 5.559 ms | 22.883 ms | 4.12× |

这些是完整第 3 层混合 QKV 的算子测试，不是完整模型生成速度。没有取得 Compute Sanitizer 结果；此前该 WSL/WDDM 环境缺少调试接口，本次没有将普通 GPU 测试结果当作内存检查通过。

定向 rustfmt、`git diff --check` 均通过。CUDA 库和 GEMV/GEMM 测试的 Clippy 检查完成；仅有既存 RMSNorm／attention／scalar 代码的 9 条告警，新增 GEMM 代码和 GPU 测试无告警。

模型装配入口现已接入：[GGUF model loader](GGUF_MODEL_LOADER.md)。该加载器保留
分段量化权重，并处理 Qwen35 的 norm、GDN 头顺序和主模型/MTP 边界。
