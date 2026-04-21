# Fused Add RMSNorm BF16 CUDA Kernel 优化实战报告

> 目标文件：`RustInfer/crates/infer-core/src/op/kernels/cuda/rmsnorm/rmsnorm.cu`
> GPU：NVIDIA A10 (SM 8.6, Ampere, 600 GB/s DRAM 峰值)
> 优化工具链：NCU (Nsight Compute) + benchmark.py 迭代循环

---

## 1. 算子分析

### 1.1 RMSNorm 公式

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

Fused 版本额外做了 residual add：

```
residual += input
norm_output = RMSNorm(residual) * weight
```

### 1.2 Kernel 计算流程

```
Pass 1: 逐元素 residual += input，同时累加 sum(x^2)
        → block-level reduction 得到 total_sum
        → 计算 inv_rms = rsqrt(total_sum / dim + eps)

Pass 2: 逐元素 norm_output = residual * inv_rms * weight
```

**核心约束**：Pass 1 和 Pass 2 之间有**数据依赖**（inv_rms 依赖全行归约结果），
无法合并为真正的单 pass，必须有一次 block 级同步。

### 1.3 内存访问模式

```
Pass 1 读：residual[row, :] + input[row, :]     → 2 × dim × 2B
Pass 1 写：residual[row, :]                      → 1 × dim × 2B
Pass 2 读：residual[row, :] + weight[:]           → 2 × dim × 2B  (residual 被第二次读取)
Pass 2 写：norm_output[row, :]                    → 1 × dim × 2B
─────────────────────────────────────────────────────
总计每行：读 4 × dim × 2B + 写 2 × dim × 2B = 6 × dim × 2B
```

对于 dim=4096：每行 48KB，128 行共 6MB → 完全适配 L2 cache (A10 有 6MB L2)。
对于 dim=8192：每行 96KB，2048 行共 192MB → DRAM bound。

---

## 2. NCU Profiling 基线诊断

### 2.1 小规模 (rows=128, dim=4096) — L2 cache 命中场景

```
GPU Speed Of Light:
  DRAM Throughput    : 53.96%
  Compute Throughput :  6.96%
  Duration           : 10.56 us

Occupancy:
  Theoretical        : 100%
  Achieved           : 28.82%      ← 严重不足
  Waves Per SM       : 0.30        ← 128 blocks / 72 SMs

Scheduler Statistics:
  No Eligible Warps  : 90.09%      ← 大量空闲
  Active Warps/Sched : 3.51 / 12

Warp Stall Analysis:
  L1TEX Stall        : 64.2%       ← 主要瓶颈
  Cycles/Issue       : 35.5        ← 每发射一条指令等 35 cycles
```

**诊断结论：Latency Bound**

- Grid 太小（128 blocks vs 72 SMs），只有 0.3 waves，无法填满 GPU
- 每行数据量小（4096 bf16 = 8KB），L2 cache 完全命中
- 实际瓶颈是 kernel launch overhead + 低 occupancy

### 2.2 大规模 (rows=2048, dim=8192) — DRAM bound 场景

```
GPU Speed Of Light:
  DRAM Throughput    : 91.34%      ← 接近饱和！
  Compute Throughput :  5.47%
  Duration           : 312.22 us

Memory Workload:
  Memory Throughput  : 547.47 GB/s (理论峰值 ~600 GB/s)
  L2 Hit Rate        : 47.93%

Occupancy:
  Achieved           : 89.91%      ← 很好
```

**诊断结论：Memory Bound (DRAM 饱和)**

- DRAM 利用率 91%，非常接近硬件极限
- 剩余优化空间 < 10%，需要从**指令效率**方向挖掘

---

## 3. 优化迭代过程

### 3.1 V1：寄存器缓存 + Warp Shuffle（失败）

**假设**：Pass 2 重新读 residual 是浪费，用寄存器缓存 Pass 1 结果可省 25% global reads。

```cuda
constexpr int MAX_CACHE = 8;
float4 cached_res[MAX_CACHE];  // 每线程 128 bytes 寄存器

// Pass 1: 缓存到寄存器
cached_res[cache_count++] = r;

// Pass 2: 从寄存器读取
r = cached_res[cache_idx++];
```

**结果**：

| 配置 | Baseline | V1 | 变化 |
|---|---|---|---|
| rows=128, dim=4096 | 0.0048ms | 0.0056ms | **-17% 变慢** |
| rows=1024, dim=4096 | 0.0721ms | 0.1051ms | **-46% 严重退化** |

**失败原因分析**：

1. **寄存器压力暴增**：每线程额外使用 8×16=128 bytes 寄存器，从 24 regs/thread 增加到 ~56
2. **Occupancy 暴跌**：更多寄存器 → 更少 blocks per SM → 更低 occupancy
3. **L2 cache 已经足够好**：Pass 1 写入的 residual 在 Pass 2 时大概率在 L2 中（temporal locality），重读代价很低

**教训**：
> 寄存器缓存并非总是优于 cache。当数据量适中且有良好的 temporal locality 时，
> L2 cache 的自然行为已经接近最优。强行用寄存器缓存反而增加压力，降低 occupancy。

### 3.2 V2：纯 Warp Shuffle + __ldg（持平）

**假设**：去掉 cub::BlockReduce 的 shared memory overhead，用 __ldg 走 read-only cache。

```cuda
// 替换 cub::BlockReduce
float warp_sum = warp_reduce_sum(sum);
__shared__ float warp_sums[8];
// ... inter-warp reduction via shared memory

// 使用 __ldg 提示 read-only
float4 inp = __ldg(&in_f4[i]);
```

**结果**：与 baseline 完全持平。

**原因**：
- `cub::BlockReduce` 内部本身就用 warp shuffle，只是多了一层抽象
- `__ldg` 在 Ampere 架构上效果有限，因为 L1 和 L2 的统一 cache 架构已经自动优化 read-only 路径
- 虽然 shared memory 从 ~260B 降到 36B，但这个量级不影响 blocks per SM

**教训**：
> 理解底层实现再做替换。cub::BlockReduce 并不"重"——它就是 warp shuffle + shared memory，
> 手写版本不会更快。__ldg 在 Volta+ 架构上已被编译器自动应用于 `const __restrict__` 指针。

### 3.3 V3：Shared Memory 缓存 Residual（持平）

**假设**：用 shared memory 缓存 Pass 1 的 residual，Pass 2 从 smem 读取避免 global re-read。

```cuda
extern __shared__ __nv_bfloat16 s_res[];  // 动态 smem = dim * 2B

// Pass 1: 写到 smem
s_res_f4[i] = r;

// Pass 2: 从 smem 读
r = s_res_f4[i];
```

**结果**：rows=1024, dim=4096 从 465 → 467 GB/s，微弱提升。

**原因**：
- dim=4096 需要 8KB smem/block，A10 每 SM 有 100KB smem
- 但 smem 容量限制 concurrent blocks per SM（从 14 降到 ~12）
- L2 cache 对 residual 的命中率已经很高（47.9% hit rate），smem 的额外延迟收益不明显

**教训**：
> Shared memory 缓存只在 L2 miss rate 高时有意义。当 L2 hit rate 足够好时，
> smem 的容量开销（减少 blocks/SM）可能抵消其延迟优势。

### 3.4 V4：FMA + bfloat1622float2 + 模板（成功✓）

**关键洞察**：既然 memory bound 已经 91%，应该从**减少指令数**的方向优化，
让每条内存事务等待期间能执行更多有用计算。

**优化点 1：FMA 指令替换 MUL+ADD**

```cuda
// 原始：2 条指令
float x0 = __bfloat162float(r_b2[j].x);
sum += x0 * x0;

// 优化：1 条 FMA 指令
float2 f2 = __bfloat1622float2(r_b2[j]);
sum = __fmaf_rn(f2.x, f2.x, sum);  // fused multiply-add
```

`__fmaf_rn(a, b, c)` = a×b+c，单条指令完成乘法和累加。
吞吐量与 FMUL 相同，但节省了单独的 FADD 指令槽。

NCU 报告中原始 kernel 有 "8832 fused and 22912 non-fused FP32 instructions"，
提示 "by converting pairs of non-fused instructions to fused equivalents,
FP32 performance could increase by up to 36%"。

**优化点 2：__bfloat1622float2 批量转换**

```cuda
// 原始：2 次独立转换
float x0 = __bfloat162float(r_b2[j].x);
float x1 = __bfloat162float(r_b2[j].y);

// 优化：1 次 pair 转换
float2 f2 = __bfloat1622float2(r_b2[j]);
```

`__bfloat1622float2` 在硬件上可能被编译为更高效的向量转换指令。

**优化点 3：Compile-time 模板 block size**

```cuda
template <int BLOCK_DIM_X>
__global__ void fused_add_rmsnorm_bf16_kernel(...) {
    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;  // 编译时常量
    // 循环步长 BLOCK_DIM_X 是编译时常量 → 编译器可以更好地 unroll 和分配寄存器
    for (int i = tid; i < vec_count; i += BLOCK_DIM_X) { ... }
}
```

用 `blockDim.x`（运行时值）vs 模板参数（编译时值）的区别：
- 编译器看到常量步长，可以精确计算循环次数，做更激进的 unroll
- `NUM_WARPS` 是 constexpr，shared memory 数组大小在编译时确定

**优化点 4：Minimal shared memory**

```cuda
// 原始 cub::BlockReduce: ~260 bytes shared memory (TempStorage)
// 优化后: 仅 36 bytes (9 floats = 8 warp sums + 1 inv_rms)
__shared__ float smem[NUM_WARPS + 1];
```

虽然 V2 中这个优化没有效果，但结合 FMA 指令减少了总执行时间后，
更紧凑的 shared memory footprint 在高占用率场景下有边际收益。

**最终结果**：

| 配置 | Baseline (GB/s) | V4 (GB/s) | 提升 |
|---|---|---|---|
| rows=32, dim=8192 | 449 | **502** | **+11.8%** |
| rows=128, dim=8192 | 896 | **982** | **+9.6%** |
| rows=512, dim=8192 | 429 | **439** | **+2.2%** |
| rows=1024, dim=8192 | 428 | **436** | **+1.8%** |
| rows=2048, dim=8192 | 420 | **429** | **+2.0%** |

dim=8192 场景稳定提升 2~12%，dim=4096 持平。

---

## 4. 关键经验总结

### 4.1 NCU 驱动优化的方法论

```
Step 1: Profile → 识别瓶颈类型
         ├── Memory Bound: DRAM Throughput > 60%, Compute < 20%
         ├── Compute Bound: Compute Throughput > 60%, DRAM < 40%
         ├── Latency Bound: 两者都低, Occupancy 低
         └── Balanced: 两者都高

Step 2: 根据瓶颈选择优化方向
         ├── Memory Bound → 减少内存事务量 (fusion, vectorization, cache)
         ├── Compute Bound → 减少指令数 (FMA, strength reduction, Tensor Core)
         ├── Latency Bound → 提高 occupancy (减少寄存器/smem, 增大 grid)
         └── Balanced → 已接近硬件极限

Step 3: 实施 → Benchmark → 重新 Profile → 验证假设
```

### 4.2 RMSNorm 类 Kernel 的性能天花板

RMSNorm 是典型的 **Memory Bound** 算子：
- 计算密度（FLOPs/Byte）极低：每个元素只做 ~5 次浮点运算
- 理论最小内存事务量 = 3 reads + 2 writes = 5 × dim × 2B/row（不可能更少）
- 实际额外开销：Pass 2 re-read residual → 多 1 次 read

当 DRAM Throughput 达到 91% 时，**只剩 ~10% 空间**。
此时优化方向只能是：
1. 减少指令数（FMA、向量化转换）→ 让内存等待期间更高效
2. 减少 overhead（warp shuffle vs block reduce）
3. 算法层面减少 pass 数（需要根本性的算法改变）

### 4.3 寄存器 vs Shared Memory vs L2 Cache 的选择

```
                延迟        带宽          容量        对 Occupancy 影响
寄存器          ~1 cycle    最高          ~256B/thread  直接影响（最敏感）
Shared Mem     ~20 cycles  ~19 TB/s      48-100KB/SM   中等影响
L1 Cache       ~30 cycles  ~19 TB/s      48-192KB/SM   无影响
L2 Cache       ~200 cycles ~4 TB/s       6MB (A10)     无影响
DRAM           ~400 cycles ~600 GB/s     24GB          无影响
```

**什么时候用寄存器缓存**：
- 每线程数据量 < 16 个 float（64B）
- 不会导致寄存器从 32 → 64+（触发 occupancy 下降）
- 数据复用次数 > 2

**什么时候用 Shared Memory 缓存**：
- 数据被 block 内多线程共享
- L2 miss rate 高（数据量 >> L2 容量）
- smem 使用量不超过 16KB/block（避免严重限制 blocks/SM）

**什么时候依赖 L2 Cache**：
- 数据有 temporal locality（写了很快就读）
- 总数据量 < L2 容量
- 不想增加寄存器/smem 压力

### 4.4 容易踩的坑

**坑 1：Warp Shuffle 广播遗漏**

```cuda
// 错误：warp 0 reduce 后的 total_sum 只存在于 warp 0 的线程中
total_sum = warp_reduce_sum(total_sum);
float inv_rms = rsqrtf(total_sum / dim + eps);  // 其他 warp 的线程拿到错误值！

// 正确：必须通过 shared memory 广播给所有线程
if (warp_id == 0 && lane_id == 0) {
    smem[0] = rsqrtf(total_sum / dim + eps);
}
__syncthreads();
float inv_rms = smem[0];  // 所有线程读到正确值
```

**坑 2：benchmark.py 的正则匹配**

`extern "C" void solve(...)` 的正则 pattern 会匹配到 kernel 函数体内的 `{`：
```
extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{
```

**解决方案**：将 `solve()` 放在文件前部（kernel 定义之前），用前向声明：
```cuda
// Forward declaration
__global__ void my_kernel(...);

// solve() 放在最前面，正则能正确匹配
extern "C" void solve(...) {
    my_kernel<<<...>>>(...);
}

// Kernel 实现在后面
__global__ void my_kernel(...) { ... }
```

**坑 3：bf16 类型在 benchmark.py 中的 workaround**

benchmark.py 只支持 `float*`, `int*` 等标准 C 类型。
使用 `short*`（同为 16 位）作为 ABI 代理：

```cuda
extern "C" void solve(short* output, const short* input, int dim) {
    my_bf16_kernel<<<...>>>(
        reinterpret_cast<__nv_bfloat16*>(output),
        reinterpret_cast<const __nv_bfloat16*>(input),
        dim
    );
}
```

Reference 中也需要对应处理：
```python
def reference(output, input, dim, **kwargs):
    out_bf16 = output.view(torch.bfloat16)  # int16 → bf16 reinterpret
    inp_bf16 = input.view(torch.bfloat16)
    # ... 计算逻辑
```

---

## 5. 完整优化后代码 diff

### 新增：`warp_reduce_sum` helper

```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

原理：`__shfl_xor_sync` 让 warp 内的线程通过 XOR mask 交换数据。
5 次迭代（offset=16,8,4,2,1）完成 32 线程的树状归约。

### 修改：`fused_add_rmsnorm_bf16_kernel`

| 位置 | 原始 | 优化后 |
|---|---|---|
| 函数签名 | `__global__ void kernel(...)` | `template<int BLOCK_DIM_X> __global__ void kernel(...)` |
| 向量化类型 | `float4*` | `uint4*`（语义更清晰） |
| bf16→f32 | `__bfloat162float` × 2 | `__bfloat1622float2` × 1 |
| 累加 | `sum += x * x` | `sum = __fmaf_rn(x, x, sum)` |
| 归约 | `cub::BlockReduce` (260B smem) | `warp_reduce_sum` (36B smem) |
| Scale 计算 | 在 Pass 2 循环内 | 在循环外预计算 |

---

## 6. 附录：常用 NCU 命令

```bash
# 快速 targeted profiling（5 个关键 section）
sudo ncu --launch-skip 5 --launch-count 1 \
    --section LaunchStats --section Occupancy --section SpeedOfLight \
    --section MemoryWorkloadAnalysis --section SchedulerStats \
    --kernel-name-base demangled -k "regex:my_kernel" \
    ./my_program

# 完整 profiling（所有 section）
sudo ncu --launch-skip 5 --launch-count 1 --set full \
    --kernel-name-base demangled -k "regex:my_kernel" \
    -o report.ncu-rep ./my_program

# 导出为文本
ncu --import report.ncu-rep --page details > report_details.txt
ncu --import report.ncu-rep --page raw > report_raw.txt
```

**关键参数说明**：
- `--launch-skip N`：跳过前 N 次 kernel launch（避免 profile warmup 阶段）
- `--launch-count 1`：只 profile 1 次 launch（减少开销）
- `-k "regex:name"`：按 kernel 名过滤
- `--set full`：收集所有 37 个 profiling pass
