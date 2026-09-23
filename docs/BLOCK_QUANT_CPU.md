# 块量化 CPU 计算

日期：2026-09-23。按用户要求，完成 CPU 后停止；CUDA 后续使用 CuTe DSL。

后续进展：量化 GPU Embedding 见 [CuTe DSL Embedding](BLOCK_QUANT_CUDA_EMBEDDING.md)。下文保留 CPU 阶段的完成边界。

## 已实现

CPU 后端纯 Rust 实现下列 13 种格式，无 llama.cpp 推理库、可执行文件或外部解码器依赖：

| 系列 | 格式 |
| --- | --- |
| 普通块 | Q8_0 |
| K 系列 | Q2_K、Q3_K、Q4_K、Q5_K、Q6_K |
| IQ2 | IQ2_XXS、IQ2_XS、IQ2_S |
| IQ3 | IQ3_XXS、IQ3_S |
| IQ4 | IQ4_NL、IQ4_XS |

入口位于 `crates/infer-backend-cpu/src/block_quant.rs`：

- `decode_block(format, bytes, out)`：单块还原为 FP32，检查输入、输出精确长度，显式读取小端字节，无对齐要求。
- `decode_row(view, row, out)`：定位并解码整行，支持行切片与每行多个块。
- CPU `MathOps::matmul_block_quant(scope, input, weight, bias, output)`：计算 `X[M,K] @ W[N,K]^T + bias`，支持 FP32、FP16、BF16。
- CPU `MathOps::embedding_block_quant(scope, weight, ids, output)`：仅解码选中 token 的行，支持重复 token 与非连续 ID 张量。

`Linear::forward` 按 `BlockQuantProjection` 中的输出段分别调用后端，保留每段编码。例如第 3 层 Q/K/V 分别为 IQ4_NL / Q4_K / Q5_K；多 token 输出按原张量 stride 写入，不把压缩字节合并为一种格式。偏置跟随输出段切片传给后端，在 FP32 中相加后才转成输出 dtype。`Embed::shared_linear` 共享同一份压缩 Storage，已能执行。

## 精度与内存契约

权重解码和点积串行累加均使用 FP32；FP16/BF16 输入先转 FP32，最终只在输出处转回原 dtype。参考路径不使用近似整数点积，也不调用原有 CPU dense matmul（其累加精度不同）。

Linear 每次只展开一行权重，临时浮点空间为 `4*K` 字节，主矩阵始终压缩存储；Embedding 同样只需要一行 FP32 暂存，以及当前 token ID 列表。例如 K=5120 时暂存约 20 KiB，K=17408 时约 68 KiB。CPU Weight 的创建仍然显式复制压缩字节；直接用 mmap `BlockQuantView` 调用 `decode_row` 则无需复制整个矩阵。

保留输入、输出、偏置的 stride 与 storage offset。输出必须与输入、压缩权重和偏置使用不同 Storage。Linear 在执行任何组合段前检查所有权重的别名；Embedding 在写入任何结果前检查所有 token ID。错误尺寸、负数/越界 ID 和不支持的整数激活 dtype 明确报错。FP16 编码中的 NaN/Inf 按浮点规则传播，不静默改为零。

块量化组件暂只支持 TP1。其他后端通过默认方法返回 `Unsupported`，没有从 CUDA 自动拷回 CPU 的隐式 fallback。

## 码表与验证边界

IQ2/IQ3 固定码表是文件编码的一部分。仓库保存了格式要求的常量、来源 commit、源码哈希、MIT 许可及只抽取常量的脚本；详见 [测试数据说明](../crates/infer-backend-cpu/tests/fixtures/block_quant/README.md)。没有运行或链接 llama.cpp，也没有将其解码结果用作测试 oracle。

测试使用独立的 Python 标准库编码脚本：先选取整数、缩放与符号并计算期望浮点结果，再打包为输入字节。13 种格式各 64 个测试块，共 **832 块**。测试编码器和 Rust 解码器共享固有码表，因而这是位布局、缩放和计算验证，不是对码表来源的独立证明。

覆盖多块、多行、符号和高位索引、各码表索引、子块缩放、half 非正规数/极值/NaN/Inf、未对齐字节、共享行切片、三种激活 dtype、非连续输出、错误尺寸和 ID，以及量化 Embedding 共享输出头、混合投影与 TP2 拒绝。CPU 库及 worker 回归共 **267 个测试通过**（832 块包含于其中一个测试内）：core 43、CPU 库 28、worker 库 178、CPU 块量化集成 7、worker 块量化集成 10、真实文件计算 1。

真实文件使用本机 `Qwen3.8-27B-UD-Q3_K_XL.gguf`：

- 506 个量化矩阵分别解码首行、中间行、末行，共 **10,383,360 个有限 FP32 数值**。
- 每种格式选一个真实张量的两行执行 Embedding 与 Linear；Linear 与 FP64 稠密点积比较，容差为 `1e-5 * sum(abs(products)) + 1e-6`。
- 取 `blk.3.attn_q/k/v.weight` 各两行、完整 K=5120，完成多 token 混合格式 Linear，对照单独解码后的 FP32 点积。此项是实际权重抽样，不是完整 attention 层前向。

复现命令：

```sh
cargo test -p infer-core -p infer-backend-cpu -p infer-worker --no-default-features --lib
cargo test -p infer-backend-cpu --test block_quant
cargo test -p infer-worker --no-default-features --test block_quant_weights
RUSTINFER_GGUF_MODEL=/root/models/Qwen3.8-27B-GGUF-Q3_K_XL/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  cargo test -p infer-worker --no-default-features --test block_quant_weights \
  block_quant_local_cpu_computation -- --ignored --nocapture
```

Clippy 检查未新增告警；worker 的 mixed_tuning 仍有已有 dead-code 告警。默认 CUDA feature 下的 `cargo check -p infer-worker --bins --tests` 也通过，验证新增接口的编译兼容性；没有执行 CUDA 量化计算。固定测试向量和 SHA256SUMS 重新生成后逐字节一致，定向 rustfmt 与 `git diff --check` 均通过。

## 当前停止点

完成的是反量化与 CPU 参考算子。尚未装配完整 GGUF 模型、接通 tokenizer 或执行文本/视觉生成；也未实现 CUDA 块量化算子。下一步 CuTe DSL 可以复用本次固定测试向量、真实权重样本、数学接口和 FP32 基准。
