# INT4 量化推理与 GEMV Kernel 优化记录

> 基于 feat/int4-quantization 分支，目标：在 RustInfer 中实现 compressed-tensors K-packed INT4 量化推理并优化 GEMV kernel 性能。

---

## 背景

- **硬件**：A10 (sm_86, 600 GB/s) 和 H20 (sm_90, 4 TB/s)
- **模型**：Llama-3.2-1B-Instruct（AWQ）、Qwen3-4B-Instruct（AWQ，MLP only）
- **量化格式**：compressed-tensors K-packed INT4

---

## 一、K-packed INT4 格式说明

compressed-tensors 的量化格式与 AutoAWQ 的 N-packed 不同：

```
weight_packed:     [N, K/8]        int32  — 每个 int32 顺序打包 8 个 K 方向的 INT4
weight_zero_point: [N/8, num_groups] int32  — zero point 沿 N 方向打包
weight_scale:      [N, num_groups]  bf16   — 每 group 一个 scale
```

反量化公式：

```
w = (extract(packed, k%8) - extract(zeros, n%8)) * scale
```

提取一个 nibble：
```c
int w = (word >> (j * 4)) & 0xF;  // j in [0, 7]
int z = (zp_packed >> (n%8 * 4)) & 0xF;
```

---

## 二、GEMV Kernel 优化迭代

GEMV（M=1, decode 阶段）是整个推理中最频繁调用的算子，Qwen3-4B 中占 GPU 总时间的 **50.4%**。

### v1：朴素版本（基线）

每个线程处理一行，串行提取 nibble：

```cuda
// 每个 int32 word，循环 8 次
for (int j = 0; j < 8; j++) {
    int w_int4 = (word >> (j * 4)) & 0xF;   // 2 条整数 ALU
    acc += ((float)w_int4 * scale + dz) * x; // int→float + FMA
}
```

**问题**：每个 word 需要 16 条整数指令（shift + mask），整数 ALU pipeline 利用率 70%，FP 利用率只有 14%。

---

### v2：FP16 Magic Number Dequant

**原理**：把 INT4 nibble 直接嵌入 FP16 的 mantissa 位，用 FP 减法代替整数提取。

```
FP16 格式：sign(1) | exponent(5) | mantissa(10)
0x6400 = 1024.0 (fp16)
0x6400 | nibble = 1024 + nibble  → 减去 1024+zero → dequant
```

实现：把一个 int32 中的 8 个 nibble 重排为 4 对 `half2`，每对包含非相邻的两个 nibble：

```cuda
static constexpr uint32_t MAGIC = 0x64006400u;  // 两个 fp16(1024.0)
static constexpr uint32_t MASK  = 0x000F000Fu;

// 4 次 shift/mask/or，得到 4 个 half2：(n0,n4), (n1,n5), (n2,n6), (n3,n7)
uint32_t p04 = ((word      ) & MASK) | MAGIC;
uint32_t p15 = ((word >>  4) & MASK) | MAGIC;
uint32_t p26 = ((word >>  8) & MASK) | MAGIC;
uint32_t p37 = ((word >> 12) & MASK) | MAGIC;

// half2 sub + mul = 4 条 FP 指令完成 8 个值的反量化
out04 = __hmul2(__hsub2(*(half2*)&p04, magic_zp), scale_h2);
```

**指令对比**（每 int32 word）：

| | 整数 ALU | FP 指令 | 总 ops |
|--|--|--|--|
| v1 朴素 | 16 | 8 FMA + 8 int2float | 32 |
| v2 FP16 magic | 12 (4 shift + 4 mask + 4 or) | 8 (half2 sub+mul) | 20 |

**效果**：ALU 利用率 70% → ~50%，A10 提升约 1.7%。

---

### v3：FP16 Magic + half2 FMA

v2 的问题：dequant 输出了 `half2`，但又拆回 float 做标量 FMA，白白浪费 half2。

改为：dequant 输出 `half2`，input 也配成 `half2`，全链路 `__hfma2`。

**配对规则**：dequant 输出 `(n0,n4), (n1,n5), (n2,n6), (n3,n7)`，input 对应配为 `(x0,x4), (x1,x5), (x2,x6), (x3,x7)`：

```cuda
// 4 × hfma2 = 8 个 FMA，4 条指令
acc_h2  = __hfma2(d04, x04, acc_h2);
acc_h2b = __hfma2(d15, x15, acc_h2b);
acc_h2  = __hfma2(d26, x26, acc_h2);
acc_h2b = __hfma2(d37, x37, acc_h2b);
```

注意两个独立的 `acc_h2` / `acc_h2b` 交替累加，避免 FMA 依赖链 stall。

**但问题**：input 是 bf16，`half2 bf16_pair_to_half2` 里仍有 `__bfloat162float` + `__float2half` = 每对 4 条转换指令。

---

### v4：BF16 Magic Number（最终版本）

**关键洞察**：input 和 scale 都是 bf16，FP16 magic 迫使所有东西绕道 fp16。直接用 BF16 做 magic number，整条链路不需要任何类型转换。

BF16 magic：

```
bf16 格式：sign(1) | exponent(8) | mantissa(7)
0x4300 = 128.0 (bf16)
0x4300 | nibble = 128 + nibble
减去 bf16(128 + zero) → dequant
```

```cuda
static constexpr uint32_t MAGIC = 0x43004300u;  // 两个 bf16(128.0)

// 提取 4 对 nv_bfloat162
uint32_t p04 = ((word      ) & MASK) | MAGIC;
...
out04 = __hmul2(__hsub2(*(__nv_bfloat162*)&p04, magic_zp), scale_bf2);

// input 直接配对为 bfloat162，零转换
__nv_bfloat162 x04 = __halves2bfloat162(inp[0], inp[4]);  // 寄存器 pack，无开销

// bf16x2 FMA
acc_a = __hfma2(d04, x04, acc_a);
```

**最终指令对比（每 int32 word，8 个 INT4）**：

| 版本 | 整数 ALU | 类型转换 | FP 指令 | 总 ops |
|------|---------|---------|---------|--------|
| v1 朴素 | 16 | 8 int2float | 8 FMA | **44** |
| v2 FP16 magic | 12 | 16 (bf16↔float↔fp16) | 8 hmul2 | **36** |
| v3 half2 FMA | 12 | 8 (bf16→fp16) | 4 hfma2 | **24** |
| v4 BF16 magic | 12 | 0 | 4 hfma2 | **20** |

---

### v5（失败）：Shared Memory 缓存 input

**思路**：block 内所有 warp 协作把 input 加载到 smem，避免 L1 cache 竞争。

**实测结果**：变慢了。

**原因**：
1. `__syncthreads()` 在 CUDA Graph 中引入同步开销
2. smem 使用量从 0 增到 K×2 bytes（8KB~28KB），每 SM 能驻留的 block 数下降，occupancy 降低
3. H20 上 L1 hit 已经 88%，L1 本身不是瓶颈，实际瓶颈是 dequant 计算

**教训**：smem 优化适合 compute-bound 且需要数据复用的场景。对 GEMV 这种每个数据用一次的 streaming 访问模式，occupancy 更重要。

---

## 三、NCU 性能分析（H20 上 kpack_gemv）

```
Kernel: kpack_gemv_kernel<4>
Grid:   (4864, 1, 1)  Block: (128, 1, 1)
CC:     9.0 (Hopper)

GPU Speed of Light:
  Compute (SM) Throughput:  71.88%   ← 计算是瓶颈
  Memory Throughput:        63.29%
  DRAM Throughput:          16.22%   ← 带宽严重未充分利用
  Duration:                 39.74 µs

Memory Analysis:
  L1/TEX Hit Rate:          88.36%
  L2 Hit Rate:               6.26%
  Memory Throughput:        651 GB/s  (H20 峰值 4 TB/s → 仅 16%)

Occupancy:
  Theoretical Occupancy:    56.25%   ← 受寄存器限制
  Achieved Occupancy:       49.43%
  Block Limit Registers:    9        ← 瓶颈

Warp State:
  Warp Cycles Per Issued Instruction: 10.30

Compute Workload Analysis:
  ALU Pipeline Utilization: 70%      ← 整数 shift/mask 填满了 ALU
  FP32 Roofline:             14%     ← 浮点严重空闲

Uncoalesced Accesses:
  43% of global load sectors wasted  ← nibble 提取导致非合并访问
```

---

## 四、根本局限：GEMV 在高带宽 GPU 上的 INT4 收益递减

| GPU | DRAM 带宽 | BF16 GEMV 瓶颈 | INT4 GEMV 状态 |
|-----|---------|--------------|--------------|
| A10 | 600 GB/s | Memory-bound | Dequant 被延迟隐藏 → 1.74x 加速 |
| H20 | 4 TB/s | Compute-bound | Dequant 暴露为瓶颈 → ~7% 加速 |

**结论**：INT4 GEMV 加速比与 `带宽/算力` 比值正相关。A10 带宽受限，INT4 4x 更小的 weight 直接换来吞吐量；H20 带宽极大，weight 一瞬间读完，时间全耗在 dequant 上。

模型越大，weight 读取时间越长，dequant 开销的相对占比越小，INT4 收益越大。**70B 模型在 H20 上 INT4 才有明显加速**。

---

## 五、架构设计

### QuantParams 抽象

```rust
pub enum QuantParams {
    Awq {
        zeros: Tensor,     // [N/8, num_groups] I32
        scales: Tensor,    // [N, num_groups] BF16
        group_size: usize,
    },
    // 未来可扩展 Gptq { ... }, Fp8 { ... }
}

pub struct Matmul {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub quant: Option<QuantParams>,
}
```

Forward dispatch：
```rust
match &self.quant {
    Some(QuantParams::Awq { zeros, scales, group_size }) => {
        if input.shape()[0] == 1 {
            kpack_gemv(...)  // decode
        } else {
            kpack_gemm(...)  // prefill
        }
    }
    None => { /* BF16/FP16/F32 路径 */ }
}
```

### RoPE Scaling

Llama3.1/3.2 的 llama3 类型 RoPE scaling 直接在 CUDA kernel 内完成，消除 CPU↔GPU 同步：

```cuda
// kernel 内判断是否需要 scaling
if (factor > 1.0f) {
    float wavelen = 2.0f * M_PI / freq;
    if (wavelen > low_freq_wavelen) {
        freq /= factor;
    } else if (wavelen > high_freq_wavelen) {
        // 线性插值
        float smooth = (original_max_pos_emb / wavelen - low_freq_factor)
                     / (high_freq_factor - low_freq_factor);
        freq = (1 - smooth) * freq / factor + smooth * freq;
    }
}
```

---

## 六、性能结果

### A10 (sm_86, 600 GB/s)

| 模型 | BF16 | INT4 AWQ | 加速比 |
|------|------|---------|--------|
| Llama-3.2-1B | 187 tok/s | **326 tok/s** | 1.74x |
| Qwen3-4B | — | **105 tok/s** | — |

### H20 (sm_90, 4 TB/s)

| 模型 | BF16 | INT4 AWQ | 加速比 |
|------|------|---------|--------|
| Llama-3.2-1B | 829 tok/s | **912 tok/s** | 1.10x |
| Qwen3-4B | 281 tok/s | **300 tok/s** | 1.07x |

---

## 七、可行的后续优化方向

### 7.1 更大模型（推荐）
8B/70B 模型 weight 读取才是真正瓶颈，INT4 能发挥 2-3x 加速。

### 7.2 Prefill GEMM 优化
当前 kpack_gemm_kernel 是朴素 per-element 实现，有 3-5x 优化空间：
- 方案 A：先 dequant 到 BF16 临时 buffer，再调 cublasLt（最简单）
- 方案 B：Shared memory tiled GEMM

### 7.3 连续批处理
Batch size > 1 时，weight 读取被多个请求复用，整体吞吐量 5-10x 提升。

### 7.4 Scale/ZeroPoint 缓存优化
当前每次 GEMV 都读 scale 和 zero_point。对于 group_size=128、K=4096，每行有 32 个 group，可以预加载到寄存器（32 个 bf16 = 64 bytes，在寄存器预算内）。
