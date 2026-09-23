# GGUF reader 实现与本地模型检查

日期：2026-09-22。

## 实现范围

`infer_worker::infrastructure::io::gguf::GgufReader` 已实现 GGUF v2/v3 小端、全部元数据类型、35 种 GGML 张量布局，以及 mmap 原始字节视图。支持逐张量混合格式，不根据文件名或 `general.file_type` 推断所有张量的编码。

```rust
use infer_worker::infrastructure::io::gguf::GgufReader;

let reader = GgufReader::open("/path/to/model.gguf")?;
for info in reader.tensors() {
    println!("{} {:?} {:?} {} bytes", info.name(), info.dimensions(),
             info.ggml_type(), info.byte_len());
}
let view = reader.read_view("output.weight")?;
// view.bytes 是绑定 reader 生命周期的压缩权重原始字节。
```

文件解析不分配 GPU 显存，不反量化，也未接入模型推理。mmap 存活期间源文件须保持不变。未知/移除的编码、越界、重叠、溢出和资源预算超限均返回结构化错误。

## 实际载入结果

使用新 Rust reader 打开 `~/models/Qwen3.8-27B-GGUF-Q3_K_XL/` 下两份文件；与固定版本上游 GGUFReader 比较了完整元数据、所有张量的名称/类型/维度/偏移/大小，并抽查每个张量首、中、尾的字节。两份文件全部一致。

| 文件 | 元数据条目 | 张量数 | 文件大小 | 数据起点 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-UD-Q3_K_XL.gguf | 50 | 866 | 13,146,393,504 B（12.24 GiB） | 10,996,640 |
| mmproj-F16.gguf | 35 | 334 | 927,607,488 B（0.864 GiB） | 20,224 |

均为 v3，alignment=32。以下大小是各组张量的编码数据之和，不包括文件头和 padding，也不是运行时显存需求。

### 语言模型

架构字段为 `qwen35`，隐藏维度 5120，FFN 中间维度 17408，词表 248320。元数据还包含完整词表、token 类型、247587 条 BPE merges 和 9993 字节聊天模板。

| 部分 | 内容 | 张量数 | 存储大小 |
| --- | --- | ---: | ---: |
| 输入 Embedding | `token_embd.weight`，Q3_K | 1 | 0.509 GiB |
| Gated DeltaNet | 48 层的 qkv/gate、卷积、状态与输出投影 | 432 | 2.566 GiB |
| 全注意力 | 16 层的 q/k/v、q/k norm、输出投影 | 96 | 0.753 GiB |
| FFN | 64 主干层的 gate/up/down | 192 | 7.262 GiB |
| 主干归一化 | 每层 attention / post-attention norm | 128 | 2.50 MiB |
| 输出层 | `output.weight`，Q5_K | 1 | 0.814 GiB |
| 最终归一化 | `output_norm.weight`，F32 | 1 | 20 KiB |
| MTP 辅助预测 | `blk.64.*`，含 attention、FFN 和 `nextn.*` | 15 | 0.327 GiB |

`qwen35.block_count=65`，同时 `qwen35.nextn_predict_layers=1`；结合 `blk.64.nextn.*` 张量，目录对应 **64 层主干 + 1 层 MTP**。后续模型适配不能直接把 65 全部当作普通主干层。此 reader 只保留这些字段和张量，不替模型层做解释或自动跳过 MTP。

主文件有 360 个 F32 张量和 506 个量化张量。量化类型共 13 种：Q2_K/Q3_K/Q4_K/Q5_K/Q6_K、Q8_0、IQ2_XXS/IQ2_XS/IQ2_S、IQ3_XXS/IQ3_S、IQ4_NL/IQ4_XS。文件名 Q3_K_XL 不代表所有张量都是 Q3_K。

### 视觉文件

架构字段为 `clip`，类型字段为 `mmproj`，projector 类型为 `qwen3vl_merger`；包含完整视觉编码器与连接语言模型的投影层。

| 部分 | 内容 | 张量数 | 存储大小 |
| --- | --- | ---: | ---: |
| Patch 输入 | 两份 patch 投影权重与 bias | 3 | 3.38 MiB |
| 位置编码 | `v.position_embd.weight` | 1 | 10.125 MiB |
| ViT 主干 | 27 层，维度 1152，16 个 attention heads | 324 | 0.767 GiB |
| Merger | `v.post_ln.*`、`mm.0.*`、`mm.2.*`，映射到 5120 维 | 6 | 0.084 GiB |

共 112 个 F16、222 个 F32 张量。图片 patch size=16，spatial merge size=2；这些是文件元数据，实际图像预处理与前向计算仍待接入验证。

## 验证与复现

- GGUF 单元测试：截断与定向变异、边界/溢出、非法元数据与形状、重叠、资源预算、嵌套数组、mmap 视图。
- 上游独立 fixture：覆盖 35 种块布局、全部元数据标量及其数组，完整对照通过。
- 两个真实文件的 ignored CPU 集成测试：均通过。
- 原有 Safetensors 读取/预取测试：3 项通过。
- CPU Clippy：完成；保留现有 `mixed_tuning.rs` 的 3 条 dead-code 警告，无 GGUF 新警告。

生成方式与命令见 [fixture README](../crates/infer-worker/tests/fixtures/gguf/README.md)。本地完整参考清单在 `target/gguf-inspection/`，不提交大模型或生成的大型清单。没有进行 GPU 推理或质量/速度评估。
