# AWQ INT4 量化推理实现总结

> RustInfer 引擎 · NVIDIA A10 GPU · Llama 3.2 1B-Instruct-AWQ

---

## 一、背景与目标

RustInfer 原本只支持 FP32/BF16 原始精度模型。为了在有限显存下运行更大模型，需要支持 AWQ (Activation-aware Weight Quantization) INT4 格式。AWQ 将每个 Linear 层的 FP16 权重压缩为 INT4（体积缩小 4 倍），是 HuggingFace 上最流行的 LLM 量化方案之一。

**最终成果**：AWQ INT4 decode 速度 **283.3 tok/s**，比 BF16 的 187.5 tok/s **快 1.51 倍**，同时权重显存占用减少约 75%。

---

## 二、AWQ 格式详解

### 2.1 数据布局

AWQ 将每个 Linear 层的 `weight [out_features, in_features]` 替换为三个张量：

```
qweight: [in_features, out_features / 8]   (int32)   — 每个 int32 打包 8 个 INT4
qzeros:  [in_features / group_size, out_features / 8] (int32) — 零点，同样打包
scales:  [in_features / group_size, out_features]     (fp16)  — 缩放因子
```

反量化公式：
```
w_fp16 = (qweight_int4 - qzeros_int4) * scales
```

`group_size` 典型值为 128，即每 128 个 in_features 共享一组 scale/zero。

### 2.2 INT4 打包顺序（AWQ_ORDER）

**这是第一个关键发现。** AutoAWQ 在打包 8 个 INT4 到一个 int32 时，**不是按顺序 [0,1,2,3,4,5,6,7] 排列的**，而是使用特殊排列：

```
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

含义：从 int32 的 bit position 0 提取的 INT4 对应 output column 0，bit position 1 对应 column 4（不是 column 1），bit position 2 对应 column 1，以此类推。

在 CUDA kernel 中提取 INT4 时必须按此顺序：

```cuda
__device__ __constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__device__ __forceinline__ int awq_extract(int packed, int pos) {
    return (packed >> (AWQ_ORDER[pos] * 4)) & 0xF;
}
```

**如果忽略 AWQ_ORDER，直接按顺序提取，模型输出完全是乱码。** 这个 bug 表现为：单个元素（pos=0）正确，但后续元素全部错位。调试时通过对比 AutoAWQ Python 库的 `dequantize_gemm()` 函数发现了 `reverse_awq_order` 这一步。

### 2.3 Llama 3.2 RoPE Scaling

**第二个关键发现。** Llama 3.2 使用特殊的 `llama3` 类型 RoPE scaling，会修改频率：

```json
{
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192
  }
}
```

对 head_dim=64 的模型，32 个频率维度中有 17 个会被修改。RustInfer 原来的 sin/cos cache 计算没有做 scaling，导致位置编码偏差。

修复方式：在 CPU 上预计算含 scaling 的 inv_freq，生成 sin/cos cache，再拷贝到 GPU。避免修改 CUDA kernel。

---

## 三、GEMV Kernel 优化历程

GEMV（矩阵向量乘法，M=1）是 decode 阶段的性能瓶颈。对于 Llama 3.2 1B，decode 一个 token 需要经过 16 层 × 4 个 GEMV = 64 次 GEMV 调用。

### 3.1 性能对比一览

| 版本 | 核心思想 | N=2048,K=2048 | N=8192,K=2048 | N=2048,K=8192 |
|------|---------|--------------|--------------|--------------|
| BF16 GEMV v3 | warp-per-row, float4 | 0.0205 ms | 0.0686 ms | 0.0686 ms |
| **v0 baseline** | 1 block/output, 标量 | 0.0512 ms | 0.2826 ms | 0.2560 ms |
| **v2 batch8** | 8-col batch/warp | 0.0236 ms | 0.0461 ms | 0.1505 ms |
| **v7 transposed** | 转置 qweight | 0.0205 ms | 0.0440 ms | 0.1260 ms |
| **v9 int4-vec** | 转置 + int4 向量化 | **0.0174 ms** | **0.0246 ms** | **0.0522 ms** |

### 3.2 v0 Baseline：为什么慢

```cuda
// 每个 block (256 threads) 只负责 1 个输出元素
// grid = N 个 blocks (N=2048 → 2048 blocks)
awq_gemv_kernel<<<N, 256>>>(...)

// 每个 thread 沿 K 循环，标量加载 qweight
for (int k = tid; k < K; k += blockDim.x) {
    int32_t qw = qweight[k * N_packed + col_packed];  // 标量读
    ...
}
```

问题：
- **1 block 只算 1 个输出**：256 个 thread reduce 到 1 个值，利用率极低
- **标量加载**：每次只读 4 bytes，无向量化
- **block 间无数据复用**：每个 block 独立读 input vector

### 3.3 v2 Batch-8：8 列共享一次权重读取

```cuda
// 每个 warp 负责 8 个输出（1 个 packed column）
// 1 次 qweight 加载服务 8 个累加器
for (int k = lane_id; k < K; k += 32) {
    int32_t qw = __ldg(&qweight[k * N_packed + col_packed]);  // 1 次读
    for (int c = 0; c < 8; c++) {
        int w_int4 = awq_extract(qw, c);   // 解包 8 个 INT4
        acc[c] += dequant(w_int4) * x[k];  // 8 个 FMA
    }
}
```

**关键洞察**：AWQ 的 1 个 int32 天然包含 8 个 INT4 权重。让 1 个 warp 同时计算这 8 个输出，qweight 只读一次，有效读取量减少 8 倍。

提升原因：
- qweight 读取量从 N 次/k 降为 N/8 次/k
- warp-level reduce 比 block-level 更高效（无 shared memory）

### 3.4 v7 Transposed：解决 non-coalesced 访问

**这是决定性的优化。**

v2 的访存分析：qweight 原始布局 `[K, N/8]`，每个 warp 遍历 K 时读 `qweight[k * N_packed + col_packed]`。一个 warp 的 32 个 lane，k 值差 1，地址差 `N_packed * 4` bytes：

```
原始布局 [K, N/8]:
  lane 0: addr = base + 0 * N_packed * 4     = base
  lane 1: addr = base + 1 * N_packed * 4     = base + 1024
  lane 2: addr = base + 2 * N_packed * 4     = base + 2048
  → 32 lanes 分散在 32KB 范围！完全 non-coalesced
```

修复：**加载时在 CPU 上转置 qweight**：`[K, N/8]` → `[N/8, K]`

```
转置后布局 [N/8, K]:
  qweight_t[col_packed, k] → 地址 = col_packed * K + k
  lane 0: addr = base + 0 * 4 = base
  lane 1: addr = base + 1 * 4 = base + 4
  lane 2: addr = base + 2 * 4 = base + 8
  → 32 lanes 连续 128 bytes！完美 coalesced
```

同理转置 qzeros 和 scales。转置只在模型加载时做一次（CPU 上几毫秒），之后所有推理都受益。

### 3.5 v9 Int4-Vectorized：匹配 BF16 的读取粒度

转置后 qweight 沿 K 方向连续，可以用 `int4`（128 bit = 4 × int32）向量化加载：

```cuda
// int4 = {x, y, z, w}，每个是 int32，对应 4 个连续 K 位置
const int4* qw_i4 = reinterpret_cast<const int4*>(qw_row);

for (int i4 = lane_id; i4 < K/4; i4 += 32) {
    int4 qw4 = __ldg(&qw_i4[i4]);  // 1 次读 128 bit = 4 × int32 = 32 个 INT4

    // 处理 4 个 K 位置，每个位置 8 个输出
    process_packed(qw4.x, qz, x[k+0], scales, acc);
    process_packed(qw4.y, qz, x[k+1], scales, acc);
    process_packed(qw4.z, qz, x[k+2], scales, acc);
    process_packed(qw4.w, qz, x[k+3], scales, acc);
}
```

**对比 BF16 GEMV**：BF16 也用 float4 加载（128 bit = 8 个 bf16）。两者读取粒度相同（128 bit/lane/load），但 AWQ 的 4 个 int32 包含 32 个 INT4 权重（对应 32 个输出通道的贡献），而 BF16 的 8 个 bf16 只对应 8 个 K 位置的贡献。AWQ 的计算密度更高。

**最终每 lane 的内循环**：
```
BF16: 每次 float4 读 16 bytes → 8 个 bf16 → 8 次 FMA
AWQ:  每次 int4  读 16 bytes → 32 个 INT4 → 对 4 个 K 位置各做 8 次 FMA = 32 次 FMA
      额外读: input (8 bytes for 4 half) + scales (少量，group 级缓存)
```

AWQ 每 16 bytes qweight 读取做 32 次 FMA，BF16 每 16 bytes weight 读取做 8 次 FMA。计算/访存比 AWQ 是 BF16 的 4 倍，这就是为什么 AWQ 能比 BF16 更快。

---

## 四、端到端集成

### 4.1 架构修改

| 文件 | 改动 |
|------|------|
| `config.rs` | 新增 `QuantizationConfig`、`RopeScalingConfig` |
| `matmul.rs` | Matmul 增加 `qzeros/scales/group_size` 字段，forward 分支 AWQ 路径 |
| `awq_gemm.cu` | 自研 GEMV/GEMM kernel |
| `awq_gemm/mod.rs` | Rust FFI 封装 |
| `llama3.rs` | AWQ 权重加载、fused QKV/GateUp 拼接、转置 |
| `inference_state.rs` | Llama 3 RoPE scaling 支持 |
| `tensor/mod.rs` | FP16 Tensor 支持 |
| 所有 .cu kernel | FP16 变体（rope/rmsnorm/embedding/swiglu/...） |

### 4.2 权重加载流程

```
safetensors 文件
    ↓ 读取 qweight [K, N/8], qzeros [G, N/8], scales [G, N]
    ↓ fused QKV: 列拼接 Q/K/V 的三组张量
    ↓ CPU 上转置: qweight [N/8, K], qzeros [N/8, G], scales [N, G]
    ↓ to_device(CUDA): 拷贝到 GPU
    ↓ 存入 Matmul 结构体
```

### 4.3 推理路径

```
Decode (M=1):
  input [1, K] (FP16)
    → awq_gemv_kernel (int4 vectorized, transposed layout)
    → output [1, N] (FP16)

Prefill (M>1):
  input [M, K] (FP16)
    → awq_gemm_kernel (naive tiled, transposed layout)
    → output [M, N] (FP16)
```

---

## 五、调试过程中的关键 Bug

### Bug 1: AWQ INT4 提取顺序错误

**现象**：模型输出完全是乱码（"aticaticatic..."）

**根因**：从 int32 中提取 INT4 时按顺序 `(packed >> (bit * 4)) & 0xF`，但 AutoAWQ 使用了特殊排列 `[0, 4, 1, 5, 2, 6, 3, 7]`

**调试方法**：用 Python AutoAWQ 库的 `dequantize_gemm()` 对比手动 dequant，发现 `reverse_awq_order` 步骤

**修复**：在 kernel 中加入 AWQ_ORDER 查找表

### Bug 2: Llama 3 RoPE Scaling 缺失

**现象**：L0 的 QKV 值与 HuggingFace 完全一致，但 attention 输出后开始偏差

**根因**：Llama 3.2 使用 `rope_type: "llama3"` 的特殊频率调整，RustInfer 的 sin/cos cache 用的是原始频率

**调试方法**：逐层打印 hidden states，与 HF `output_hidden_states` 对比，发现 RoPE 后开始分歧；进一步对比 sin/cos 频率发现 17/32 个频率不同

**修复**：解析 `rope_scaling` config，在 CPU 上预算含 scaling 的 sin/cos cache

### Bug 3: Fused QKV 拼接方向错误

**现象**：gate_up 输出含 -inf/NaN

**根因**：qweight `[K, N/8]` 的 fuse 应该沿列（N）方向拼接，但代码用了行（K）方向拼接

**修复**：实现 `fuse_tensors_vertically_cols` 函数做列拼接

---

## 六、性能总结

### Kernel 级

| Shape | BF16 GEMV | AWQ v9 | 加速比 |
|-------|-----------|--------|--------|
| N=2048, K=2048 (attn) | 0.0205 ms | 0.0174 ms | **1.18x** |
| N=8192, K=2048 (ffn_gate) | 0.0686 ms | 0.0246 ms | **2.79x** |
| N=2048, K=8192 (ffn_down) | 0.0686 ms | 0.0522 ms | **1.31x** |

### 端到端

| 指标 | BF16 | AWQ INT4 |
|------|------|----------|
| Decode 速度 | 187.5 tok/s | **283.3 tok/s** |
| 加速比 | 1.00x | **1.51x** |
| 权重显存 (1B 模型) | ~2 GB | ~0.5 GB |

### 优化演进

```
v0 (naive)          48 tok/s   ████
v2 (batch-8)       156 tok/s   ████████████████
v7 (transposed)    169 tok/s   █████████████████
v9 (int4-vec)      283 tok/s   █████████████████████████████
BF16 (reference)   187 tok/s   ███████████████████
```

---

## 七、关键经验

1. **量化格式有坑**：AWQ 的 INT4 打包顺序不是直觉上的顺序排列，必须看实际库的源码确认。

2. **转置是最有效的优化**：GEMV 的性能由访存模式决定。qweight 的原始布局 `[K, N/8]` 对 GEMV 完全不友好，转置为 `[N/8, K]` 后带宽利用率从 15% 提升到 60%+。这个优化零额外开销（只在加载时做一次）。

3. **向量化加载匹配硬件粒度**：GPU 的内存控制器以 32/128 byte sector 为单位服务请求。int4（128 bit）向量化加载让每次请求读满一个 sector，避免浪费带宽。

4. **INT4 的真正优势是计算密度**：同样 128 bit 的读取，BF16 得到 8 个权重做 8 次 FMA，INT4 得到 32 个权重做 32 次 FMA（分配到 8 个输出通道各 4 次）。这使得 AWQ 在带宽相同时做更多有效计算。

5. **调试量化模型靠逐层对比**：用 HuggingFace 的参考实现做逐层 hidden state 对比，精确到浮点值级别，是定位 bug 最有效的方法。
