# CuTe DSL 块量化 Embedding

本文记录 GPU Embedding 阶段。后续已接通单 token 的[量化 GEMV](BLOCK_QUANT_CUDA_GEMV.md)；[多 token GEMM](BLOCK_QUANT_CUDA_GEMM.md) 也已接通。

## 调用路径

```text
Embed::forward(BlockQuant)
  → Cuda::embedding_block_quant
  → 校验形状、设备/配置、dtype、stride、别名、token ID
  → CuTe DSL AOT 内核：读取压缩块 → FP32 解码 → 转换并写入输出
```

支持 Q8_0、Q2_K/Q3_K/Q4_K/Q5_K/Q6_K、IQ2_XXS/IQ2_XS/IQ2_S、
IQ3_XXS/IQ3_S、IQ4_NL/IQ4_XS，共 13 种格式；每种支持 FP32、FP16、BF16
输出，共 39 个 AOT 变体。无 Python 或 llama.cpp 运行依赖。

沿用 `cute-dsl` Cargo feature 与固定的 `nvidia-cutlass-dsl==4.7.1` 构建环境：

```sh
export RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python"
export PATH="$PWD/.venv/bin:$PATH"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
cargo build -p infer-worker --features cute-dsl
```

未开启 feature、或编译目标与实际 GPU 架构不匹配时，量化 Embedding 返回
`Unsupported`，不静默退回 CPU。已有稠密 Embedding 保持原路径。

## 内核与存储

一 CTA 处理一个选中 token 的一个量化块，一线程产生一个权重元素。
Q8_0/IQ4_NL 使用 32 线程，其余格式使用 256 线程。只读取所需 token 的
压缩行，不展开完整浮点 Embedding 表。

权重、行切片和输出地址使用 64 位运算；支持非连续 ID、重复 ID、广播 ID、
输出行/列 stride、转置输出及非零 storage offset。输出元素不能互相重叠，
也不能与 ID 或权重共享 Storage。所有位读取明确按小端组合，不要求块地址
按 FP16 对齐。CuTe DSL 的窄整数读取在扩展后显式屏蔽高位，避免符号扩展
污染压缩字节。

IQ2/IQ3 码表集中到 `infer-core/src/dtype/quant/codebooks.rs`，CPU 直接引用，
CUDA 构建时抽取同一组常量，避免维护两套码表。GPU 表包含 IQ4 的 16 个
非线性值，总计 17,424 字节，每个 CUDA 配置创建时上传一次。来源与 MIT
许可保留在同目录。GPU 模块提前加载，随配置生命周期释放。

编译器校验 PTX 入口、10 个参数的 ABI、线程数、架构及共享内存要求，
生成 Rust 清单。解码使用 FP32 算术，禁用 ptxas 的 FMA 合并，输出时才转
FP16/BF16，便于与 CPU 参考逐元素比较。

## 检查与当前限制

现有接口要求非法 token 在写入任何输出前报错。因此第一版在实际执行流上
将 ID（不是权重）拷贝到 pinned host buffer，等待复制完成并检查整批 ID，
然后异步提交 GPU 解码内核。没有 CPU 反量化或浮点权重上传。

这会引入每次调用的 host buffer 分配和一次 stream 同步，暂不适合 CUDA
Graph 捕获；捕获期间在分配/复制/写出之前明确返回 `Unsupported`，捕获可
正常中止后恢复 eager 执行。后续如需优化，应设计已验证 ID 的异步接口或
设备错误状态处理，不能直接删掉越界检查改变错误契约。

CPU/GPU 接口都仍限制块量化组件为 TP1。此阶段不表示完整 GGUF 模型能生成。

## 验证入口

```sh
cargo test -p infer-backend-cuda --features cute-dsl \
  --test cute_block_embedding -- --ignored --test-threads=1

RUSTINFER_GGUF_MODEL=/root/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  cargo test -p infer-worker --features cute-dsl --test block_quant_weights \
  block_quant_local_cuda_embedding -- --ignored --nocapture --test-threads=1
```

固定向量沿用 CPU 阶段的 832 个测试块，GPU 覆盖全部 13 种格式及三种输出
dtype。额外检查多块/行切片、ID/output stride、边界哨兵、非法输入、配置隔离、
奇数地址的压缩权重、捕获拒绝和恢复。真实文件测试将完整 `[248320,5120]` Q3_K token embedding
表上传 GPU，通过实际 `Embed::forward` 查询普通/特殊/末尾/重复 token，
检查 FP32 与 BF16 输出。

RTX 4070 Ti Super 上的 5 个 GPU 集成测试已通过：三种输出 dtype 与 CPU
逐元素一致，配置隔离和错误路径也通过。真实文件的完整 Q3_K 表
（546,304,000 字节，约 521 MiB）也已上传并通过 `Embed::forward` 验证：
7 个 token 的 FP32/BF16 输出与 CPU 完全一致。另尝试了 Compute Sanitizer；当前
WSL/WDDM 环境未启用调试接口，工具报告无法初始化并不支持该设备，未能
完成内存检查。因此不能把普通 GPU 测试通过视为 memcheck 通过。

回归：原有 CuTe RMSNorm 的 9 个 GPU 测试、core/CPU/worker 的 266 个 CPU
测试均通过；共享码表移动后，832 个固定测试块及 SHA256SUMS 重新生成后
逐字节一致。代码通过定向 rustfmt 与 `git diff --check`。
CuTe feature 下的 Clippy 检查也完成；9 条告警来自已有 RMSNorm、attention
和 scalar 路径，没有新增 Embedding 告警。

## 进展提交的隔离验证

本次提交只包含 GGUF 解析、块量化权重表示、CPU 算子、CuTe Embedding 及其
必要调用点迁移。开发工作区中的 AWQ 扩展、调度、前端及 attention 实验没有
纳入，因此上文开发工作区的回归总数与提交快照的总数不同。

使用与开发工作区一致的 `cargo +stable`，从暂存区导出的独立快照运行
`cargo test --locked -p infer-core -p infer-backend-cpu -p infer-worker
--no-default-features --lib --test block_quant --test block_quant_weights --test gguf`：
**257 个测试通过**（core 43、CPU 库 28、worker 库 168、块量化集成 17、GGUF 1）。
同一快照的 `cargo +stable check --locked -p infer-worker --features cute-dsl
--bins --tests` 通过；仅有已有的 `PREFILL_FA3_ENV` dead-code 告警。
