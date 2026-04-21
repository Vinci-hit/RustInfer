# BF16 GEMV 算子优化实战总结

> 目标算子: `hgemv_bf16` — BF16 矩阵向量乘 (decode phase, M=1)
> 硬件平台: NVIDIA A10 (Ampere, sm_86, 72 SMs, ~600 GB/s DRAM BW)
> 问题规模: N=11008, K=4096 (LLaMA-7B FFN up/gate projection)
> 优化轮数: 3 轮迭代

---

## 1. 算子背景

### 1.1 什么是 GEMV

GEMV = General Matrix-Vector Multiplication，即 **y = W · x**：
- W: 权重矩阵 `[N, K]`，row-major 存储
- x: 输入向量 `[K]`
- y: 输出向量 `[N]`

在 LLM 推理的 **decode 阶段**（逐 token 生成），batch size = 1，所有线性层退化为 GEMV。
此时 cuBLAS 的启动开销太大，手写 kernel 通常比 cuBLAS 快 1.5x 以上。

### 1.2 为什么 GEMV 是 Memory Bound

GEMV 的算术强度（Arithmetic Intensity）极低：
```
计算量: 2 * N * K FLOPs
访存量: (N * K + K + N) * sizeof(bf16)  ≈ N * K * 2 bytes

AI = 2*N*K / (N*K*2) = 1 FLOP/byte
```
A10 的算力 ~62 TFLOPS (FP32)，带宽 ~600 GB/s，算力/带宽比 = 103 FLOP/byte。
AI = 1 远小于 103，因此 **GEMV 100% 是 memory bound**，性能上限由 DRAM 带宽决定。

理论最优延迟:
```
数据量 = (11008 * 4096 + 4096 + 11008) * 2 bytes = 90.2 MB
理论下限 = 90.2 MB / 600 GB/s ≈ 0.150 ms
```

---

## 2. Baseline 分析

### 2.1 Baseline 内核设计

```
┌─────────────────────────────────────────────────┐
│  Block (256 threads = 8 warps)                  │
│                                                 │
│  ┌──────────────────────────────────────┐       │
│  │ Shared Memory: input vector x [K]    │ ← 所有线程协作加载 input 到 smem
│  └──────────────────────────────────────┘       │
│        __syncthreads()                          │
│                                                 │
│  Warp 0 → row 0:  dot(W[0,:], smem_x)          │
│  Warp 1 → row 1:  dot(W[1,:], smem_x)          │
│  ...                                            │
│  Warp 7 → row 7:  dot(W[7,:], smem_x)          │
│                                                 │
│  每个 warp 内: 32 线程并行, warp shuffle reduce  │
└─────────────────────────────────────────────────┘
```

关键参数:
- 1 warp = 1 row，1 block = 8 warps = 8 rows
- 向量化: float4 (128-bit) 一次加载 8 个 bf16
- 累加: FP32 精度（避免 bf16 累加的精度损失）
- Reduce: warp shuffle（`__shfl_xor_sync`），无需 shared memory

### 2.2 Baseline NCU Profiling 关键指标

```
┌─────────────────────────────────┬──────────┬────────────────────────────────────────┐
│ 指标                            │ 值       │ 解读                                   │
├─────────────────────────────────┼──────────┼────────────────────────────────────────┤
│ Duration                        │ 185 μs   │ 基线延迟                               │
│ DRAM Throughput                  │ 92.6%    │ 非常高，接近硬件极限                   │
│ Memory Bandwidth                │ 555 GB/s │ A10 理论 ~600 GB/s                     │
│ Compute (SM) Throughput          │ 14.6%    │ 极低 → 典型 memory bound               │
│ SM Busy                          │ 15.0%    │ 计算资源严重空闲                       │
│ Achieved Occupancy               │ 87.0%    │ 不错                                   │
│ Registers Per Thread             │ 38       │ 中等                                   │
│ L1/TEX Hit Rate                  │ 8.3%     │ L1 缓存几乎没用上                      │
│ No Eligible Warps                │ 86.7%    │ 绝大部分时间 scheduler 找不到可执行 warp │
│ Executed IPC                     │ 0.53     │ 很低（理论最高 ~4）                    │
│ Shared Memory (dynamic)         │ 8.19 KB  │ 存 input vector                        │
└─────────────────────────────────┴──────────┴────────────────────────────────────────┘
```

### 2.3 瓶颈诊断流程

NCU 分析的标准诊断顺序:

```
Step 1: SpeedOfLight → 确定大类
  DRAM Throughput = 92.6% >> SM Throughput = 14.6%
  ⟹ DRAM Memory Bound（不是 compute bound，不是 latency bound）

Step 2: MemoryWorkloadAnalysis → 定位访存细节
  L1/TEX Hit = 8.3%  → input vector 每次都从 DRAM 走，虽然存了 smem
  L2 Hit = 3.68%     → weight matrix 太大（~86MB），L2 只有 6MB，几乎全 miss
  Mem Busy = 48.74%  → 内存控制器忙碌度中等

Step 3: Occupancy → 并行度够不够
  Achieved = 87%, Theoretical = 100%
  Block Limit: Registers = 6, Warps = 6, Shared Mem = 7
  ⟹ Register 和 warp 数同时限制 occupancy

Step 4: SchedulerStats → 延迟隐藏效果
  No Eligible = 86.7%
  ⟹ 尽管 occupancy 87%，但大部分 warp 都在等待内存响应
     这是 memory bound kernel 的典型表现
```

---

## 3. 优化迭代

### 3.1 V1: 多行复用 (REJECTED ❌)

**假设**: 每个 warp 处理 4 行而不是 1 行，可以让 input vector 的读取在更多行之间分摊。

```
                    V0: 1 row/warp                V1: 4 rows/warp
                ┌───────────────────┐         ┌───────────────────┐
   Warp 0       │ load x, load W[0] │         │ load x            │
                │ dot(W[0], x)      │         │ load W[0]..W[3]   │
                └───────────────────┘         │ dot(W[0..3], x)   │
                                              └───────────────────┘
   寄存器需求    │  1 个累加器 sum    │         │ 4 个累加器 sums[] │
                │  + x_pack, w_pack │         │ + x_pack          │
                │                   │         │ + 4 个 w_pack      │
```

**NCU 结果**:
```
                    V0          V1
Registers/Thread    38          75  ← 翻倍！
Theoretical Occ.   100%         50% ← 腰斩！
Achieved Occ.       87%         42%
Duration           185 μs      194 μs ← 更慢了！
DRAM Throughput    92.6%       91.7%
```

**教训**:
> 在已经 92%+ DRAM 带宽利用的 memory bound kernel 上，增加 "数据复用" 的收益极其有限。
> 而多行复用带来的 register pressure 上升会直接砍 occupancy，减少可并行的 warp 数。
> 对 memory bound kernel，occupancy 下降 = 延迟隐藏能力下降 = 性能退化。

**策略记忆: REJECTED** — `multi-row-per-warp` 在高带宽利用的 GEMV 中不适用。

---

### 3.2 V2: 去掉 Shared Memory (POSITIVE ✅)

**核心洞察**: Baseline 用 shared memory 存 input vector (8KB)。但 A10 的 L1 cache = 128KB。
input vector 只有 8KB，第一个 warp 从 DRAM 读入后就会自动缓存在 L1，后续所有 warp 直接从 L1 读取。
shared memory 反而引入了额外开销:
- 需要 `__syncthreads()` 同步（全 block 256 个线程等待）
- 占用 shared memory 配额（可能影响 occupancy）
- 需要额外的 cooperative load 代码

```
          V0 (smem)                          V2 (L1 cache)
   ┌──────────────────┐              ┌──────────────────┐
   │ All threads:     │              │                  │
   │   smem[i] = x[i] │              │ (no setup)       │
   │ __syncthreads()  │ ← 开销!     │                  │
   │                  │              │                  │
   │ Warp 0:          │              │ Warp 0:          │
   │   x = smem[i]    │              │   x = __ldg(x+i) │ ← L1 miss → DRAM
   │   w = __ldg(W)   │              │   w = __ldg(W+i) │
   │                  │              │                  │
   │ Warp 1:          │              │ Warp 1:          │
   │   x = smem[i]    │              │   x = __ldg(x+i) │ ← L1 HIT! (缓存命中)
   │   w = __ldg(W)   │              │   w = __ldg(W+i) │
   └──────────────────┘              └──────────────────┘
```

**NCU 结果**:
```
                    V0          V2
L1/TEX Hit Rate     8.3%       49.7%  ← 飙升！input 被 L1 缓存
Duration           185 μs      181 μs
DRAM Throughput    92.6%       93.9%
Registers/Thread    38          42    ← 略增（__ldg 指令的影响）
Achieved Occ.       87%         76%   ← register 42 导致下降
```

**有效但还有改进空间**: register 从 38 涨到 42 导致 occupancy 下降。

---

### 3.3 V3: 精细调优 Register + Launch Bounds (BEST ✅)

在 V2 基础上优化:

**改动 1: `__launch_bounds__` 控制 occupancy**
```cuda
// 告诉编译器: 每 block 256 线程，每 SM 至少 6 个 block
__launch_bounds__(WARPS_PER_BLOCK * 32, 6)
```
这让编译器更积极地将变量放入寄存器重用，而不是分配新寄存器。

**改动 2: 手动展开 warp shuffle**
```cuda
// V0: 循环版本（编译器可能保留 loop counter 寄存器）
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
}

// V3: 完全手动展开（消除 offset 变量）
val += __shfl_xor_sync(0xffffffff, val, 16);
val += __shfl_xor_sync(0xffffffff, val, 8);
val += __shfl_xor_sync(0xffffffff, val, 4);
val += __shfl_xor_sync(0xffffffff, val, 2);
val += __shfl_xor_sync(0xffffffff, val, 1);
```

**改动 3: 位运算替代除法**
```cuda
// V0
const int warp_id = tid / 32;
const int lane_id = tid % 32;

// V3 (编译器通常会做，但显式写更安全)
const int lane_id = threadIdx.x & 31;
const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
```

**最终 NCU 结果**:
```
                    V0          V3          变化
Duration           185 μs      181.8 μs    -1.7%
DRAM Throughput    92.6%       93.9%       +1.3pp
Memory BW          555 GB/s    563 GB/s    +1.4%
L1/TEX Hit Rate     8.3%       49.8%       +41.5pp
Registers/Thread    38          40          +2
Theoretical Occ.   100%        100%        =
Achieved Occ.       87%         91.6%      +4.6pp
Instructions       5,911K      5,713K      -3.3%
```

---

## 4. 全版本对比

```
┌──────────┬────────────┬─────────┬──────────┬───────────┬──────┬───────────────────┐
│ 版本     │ Median(ms) │ BW(GB/s)│ DRAM 利用│ Occupancy │ Regs │ 结论              │
├──────────┼────────────┼─────────┼──────────┼───────────┼──────┼───────────────────┤
│ V0 基线  │   0.1823   │   555   │  92.6%   │   87.0%   │  38  │ 基线              │
│ V1       │   0.1902   │   486   │  91.7%   │   42.0%   │  75  │ ❌ rejected       │
│ V2       │   0.1802   │   563   │  93.9%   │   76.5%   │  42  │ ✅ positive       │
│ V3 ★    │   0.1802   │   563   │  93.9%   │   91.6%   │  40  │ ✅ best (positive)│
└──────────┴────────────┴─────────┴──────────┴───────────┴──────┴───────────────────┘

理论带宽极限:  ~600 GB/s
V3 实际带宽:   563 GB/s = 93.9% 利用率
理论最低延迟:  90.2 MB / 600 GB/s = 0.150 ms
V3 实际延迟:   0.180 ms = 理论的 1.20x
```

---

## 5. 关键学习要点

### 5.1 Memory Bound 优化的核心原则

```
                    优化优先级

                ┌─────────────────────┐
  最高优先级 →  │ 1. 减少总访存量     │  ← 唯一能突破带宽上限的方法
                │    (量化/剪枝/融合) │     但本次只做 kernel 级优化
                ├─────────────────────┤
  高优先级   →  │ 2. 提高带宽利用率   │  ← 对齐、向量化、coalesced access
                │    (已经 92.6%)     │     本次已接近极限
                ├─────────────────────┤
  中优先级   →  │ 3. 减少开销         │  ← 去 syncthreads、减指令数
                │    (V3 的方向)      │     本次收益 ~2%
                ├─────────────────────┤
  低优先级   →  │ 4. 提高 ILP/OCC     │  ← 调 occupancy、launch_bounds
                │    (V3 收尾)        │     Memory bound 时收益有限
                └─────────────────────┘
```

### 5.2 Shared Memory vs L1 Cache 的选择

| 场景 | 用 Shared Memory | 用 L1 Cache |
|---|---|---|
| 数据需要跨 warp 精确共享 | ✅ | ❌ |
| 数据量远大于 L1 cache | ✅ | ❌ |
| 数据量小 (< 32KB) 且只读 | ❌ | ✅ ← GEMV 的 input |
| 需要避免 `__syncthreads()` | ❌ | ✅ |
| 访问模式不规则(bank conflict) | 需要 padding | 自动处理 |

**本次核心发现**: 对于 GEMV，input vector 只有 8KB (K=4096)，远小于 L1 cache 的 128KB。
用 `__ldg()` 显式走 read-only cache 路径，L1 命中率从 8% 飙升到 50%，
同时省掉了 `__syncthreads()` 和 cooperative load 的开销。

### 5.3 Register Pressure 的蝴蝶效应

```
更多 live variables
    → 更多 registers/thread
        → 更少 blocks/SM (block limit by registers)
            → 更低 occupancy
                → 更少 active warps
                    → 更差的 latency hiding
                        → 性能下降

V1 的 register: 38 → 75
V1 的 occupancy: 87% → 42%
V1 的结果: 更慢！
```

控制 register 的手段:
1. `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)`
2. `-maxrregcount=N` 编译选项
3. 减少 live variables（避免大数组、减少循环展开深度）
4. 用 `#pragma unroll 1` 禁止某些循环的展开

### 5.4 NCU Profiling 的诊断地图

```
Start
  │
  ▼
SpeedOfLight Section
  │
  ├── DRAM% >> SM% ──────────► MEMORY BOUND
  │                              │
  │                              ├─ L1 Hit 低 → 改善 cache 利用
  │                              ├─ coalescing 差 → 改善访存对齐
  │                              ├─ 向量化不足 → 用 float4/LDG.128
  │                              └─ 总访存量大 → fusion / 量化 / tiling
  │
  ├── SM% >> DRAM% ──────────► COMPUTE BOUND
  │                              │
  │                              ├─ 没用 Tensor Core → 用 MMA/WMMA
  │                              ├─ FMA 利用低 → 改善 ILP
  │                              └─ 分支多 → 消除 warp divergence
  │
  └── 都 < 40% ─────────────► LATENCY BOUND
                                 │
                                 ├─ Occupancy 低 → 减 reg / smem / 调 block size
                                 ├─ No Eligible 高 → 增加 ILP / pipeline
                                 └─ 同步太多 → 减少 syncthreads / 用 syncwarp
```

### 5.5 "已经很好" 时的优化心态

当 baseline 已经达到 92%+ DRAM 带宽利用时:
- ❌ 不要期待 2x 的加速
- ❌ 不要为了"看起来高级"而堆技巧
- ✅ 聚焦减少固定开销（同步、指令数、launch overhead）
- ✅ 2-5% 的改善就是很好的结果
- ✅ 考虑 kernel fusion（和前后算子合并）才能有质变

---

## 6. 延伸: 如果要进一步优化

| 方向 | 预期收益 | 复杂度 | 说明 |
|---|---|---|---|
| INT4/INT8 权重量化 | 2-4x | 高 | 将 W 从 bf16(2B) 量化到 int4(0.5B)，访存量减少 4x |
| Kernel Fusion | 10-30% | 中 | 将 GEMV + bias + activation 合并为一个 kernel |
| Persistent Kernel | 5-10% | 中 | 减少 kernel launch overhead（对小 N 有效）|
| L2 Persistence | 3-5% | 低 | 用 `cudaAccessPolicyWindow` 钉住 input vector |
| Split-K | 视情况 | 中 | 当 K 很大时，多个 warp 分段处理同一行再 reduce |

---

## 附录: 完整优化后的 V3 内核

```cuda
// BF16 GEMV v3: no shared memory, L1 cache input, __launch_bounds__
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
hgemv_bf16_v3_kernel(
    const __nv_bfloat16* __restrict__ input,   // [K]
    const __nv_bfloat16* __restrict__ weight,  // [N, K]
    __nv_bfloat16* __restrict__ output,        // [N]
    const int N, const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;

    const int pack_num = K >> 3;
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* weight_f4 = reinterpret_cast<const float4*>(weight + row * K);

    float sum = 0.0f;
    for (int i = lane_id; i < pack_num; i += 32) {
        float4 x = __ldg(input_f4 + i);
        float4 w = __ldg(weight_f4 + i);
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&x);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w);
        sum += __bfloat162float(xb[0]) * __bfloat162float(wb[0]);
        sum += __bfloat162float(xb[1]) * __bfloat162float(wb[1]);
        sum += __bfloat162float(xb[2]) * __bfloat162float(wb[2]);
        sum += __bfloat162float(xb[3]) * __bfloat162float(wb[3]);
        sum += __bfloat162float(xb[4]) * __bfloat162float(wb[4]);
        sum += __bfloat162float(xb[5]) * __bfloat162float(wb[5]);
        sum += __bfloat162float(xb[6]) * __bfloat162float(wb[6]);
        sum += __bfloat162float(xb[7]) * __bfloat162float(wb[7]);
    }

    // Warp shuffle reduce (fully unrolled, no loop variable)
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    if (lane_id == 0)
        output[row] = __float2bfloat16(sum);
}
```
