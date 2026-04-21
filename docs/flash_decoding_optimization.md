# Flash Decoding BF16 (head_dim=128) 优化总结报告

> 目标算子：`RustInfer/crates/infer-core/src/op/kernels/cuda/flash_attn_gqa/flash_decoding_bf16_hdim128.cu`
> 硬件：NVIDIA A10 (SM_86, 72 SMs, ~600 GB/s peak DRAM)
> 测试 shape：`num_q_heads=32, num_kv_heads=8, head_dim=128, seq_len=2048`（decode 阶段，`q_seq_len=1`）
> 工具链：CUDA 12.4 + Nsight Compute 2024.1（需 `sudo -n ncu` 绕过 `RmProfilingAdminOnly=1`）
> 方法：`cuda-optimized-skill/skills/optimized-skill/operator-optimize-loop`，4 轮迭代
> 日期：2026-04-21

---

## 一、最终结果

| 版本 | Median (ms) | Speedup vs v0 | Correctness | Full NCU |
|---|---:|---:|---|---|
| v0 baseline（原版） | 0.0589 | 1.00× | PASS | ✓ |
| v1 split-K（N_SPLIT=8） | 0.0303 | **1.94×** | PASS | ✓ |
| **v2 split-K + bf16 s_acc** | **0.0255** | **2.31×** | **PASS** | ✓ ← **winner** |
| v3 launch_bounds + FMA | 0.0258 | 2.28× | PASS | ✓ |

- **端到端 kernel median 从 58.9 μs 降到 25.5 μs，提速 2.31×**。
- 对 PyTorch bf16-quantised reference 的 speedup：`~18×`（reference 用 `torch.softmax/einsum` fp32 重算）。
- Correctness 每轮 `max |delta| ≈ 4.88e-4`，稳定在 bf16 精度范围内。

---

## 二、迭代方法论

该 skill 要求每轮都必须跑完：

```
correctness → benchmark → targeted NCU → full NCU → proposal → 下一版算子
```

每轮的产物固定落盘到：

```
optimize_runs/run_main/iter_vN/
  ├── v*_vN.cu                     # 本轮算子快照
  ├── benchmark_result.json        # 结构化 benchmark 指标
  ├── targeted.ncu-rep             # NCU LaunchStats / Occupancy / SoL / Mem / Scheduler
  ├── full.ncu-rep                 # `--set full` 完整 NCU 报告
  ├── targeted_summary.txt         # ncu --import 导出的人类可读摘要
  ├── full_summary.txt
  ├── iteration_summary.md         # 本轮 benchmark+NCU 落点
  └── optimization_proposal.md     # 含 ## Strategy tags 的本轮策略
```

**铁律**：最终宣称最佳版本前必须满足
- benchmark 成功
- correctness 通过（若提供 reference）
- full `.ncu-rep` 存在且 `> 0 bytes`

---

## 三、各轮优化详解

### v0：基线（原版算子）

**代码结构**：
- 单 kernel，一个 block 对应一个 query head，每 block 256 线程 = 16 组 × 16 lane。
- 每 block 串行扫完 2048 tokens，每 16 tokens 一个 micro-tile，双 buffer `cp.async` 流水线。
- Group-merge 用 fp32 `s_acc[16 × 128]` = 8 KB。

**v0 NCU 快照**：

| 指标 | 值 | 诊断 |
|---|---:|---|
| Duration | 101.3 μs | — |
| Grid | 32 blocks | **严重欠载** |
| Waves/SM | **0.15** | **40+ 个 SM 全程空转** |
| Theoretical Occupancy | 50 % | SMEM 卡到 3 blocks/SM |
| **Achieved Occupancy** | **16.6 %** | — |
| SM Busy | 39 % | — |
| DRAM Throughput | 19.5 % | 没接近带宽天花板 |
| L2 Hit Rate | 74.9 % | GQA group 已自然命中 L2 |
| No Eligible | 61 % | latency-bound |
| Warp Cycles/Issued | 5.1 | — |

**一级判定（按 `optim.md` 的 Roofline 规则）**：
- SM SOL 17%、Memory SOL 25% 都很低 → **Latency Bound**（不是 memory-bound，也不是 compute-bound）。

**根因**：`grid = num_q_heads = 32` 在 72 个 SM 上直接把 GPU 空掉一半。任何带宽/算力指标都只是这个现象的派生。

---

### v1：Split-K over KV sequence （1.94×）

**策略来源**：
- `reference/cuda/optim.md` §3.3 Latency Bound → 提升 Occupancy → "增加每线程独立工作量 / 增加 grid 并行度"。
- FlashDecoding / FlashInfer / vLLM decode 阶段的工业标准做法。

**核心思想**：把 seq_len 维度切成 `N_SPLIT=8` 段，每段独立做 online softmax 得到部分 `(m_i, l_i, O_i)`，再用 log-sum-exp 合并：

```
grid_pass1 = (num_q_heads, N_SPLIT) = 32 × 8 = 256 blocks
grid_pass2 = (num_q_heads,)           = 32 blocks（纯 LSE 合并，极小）
```

**关键实现细节**：
- `O_g[q, s, d]` 存**未归一化**的 partial accumulator（`block_acc`），不要除以 `block_sum`——pass 2 需要原始 `(m, l, o)` 才能正确合并。
- 空 chunk 需要写入 `(m=-inf, l=0, O=0)` 作为合并单位元，否则 `expf(-inf - g_m) * anything = 0` 也能工作，但显式写更稳。
- `chunk_size = ceil(seq_len / N_SPLIT)` 向上取整到 `BN128=16` 的倍数，保证 `cp.async` 的 16-token tile 对齐。

**v1 NCU**：

| 指标 | v0 | v1 | Δ |
|---|---:|---:|---:|
| Grid | 32 | 256 | 8× |
| Waves/SM | 0.15 | 1.19 | 7.9× |
| Achieved Occupancy | 16.6 % | 43.5 % | 2.6× |
| DRAM Throughput | 19.5 % | 41.9 % | 2.1× |
| Duration μs | 101.3 | 47.0 | -54% |
| **kernel median ms** | **0.0589** | **0.0303** | **1.94×** |

**新瓶颈**：`Block Limit SMEM = 3`，SMEM 已成为 occupancy 天花板。

---

### v2：bf16 s_acc 降压 SMEM （再 ×1.19，累计 2.31×）

**策略来源**：`reference/cuda/memory-optim.md` §二 "Shared Memory 优化" + §五 "寄存器与压力控制"。

**核心**：`s_acc` 在 v1 里是 16 组 × 128 dim × **fp32** = 8 KB，占单 block SMEM 32%。这块只在块内最后一次 group-merge 用一次，完全没必要 fp32 存储。

```cpp
// v1: float s_acc[NUM_GROUPS128 * HD128];   // 8 KB
__nv_bfloat16 s_acc_bf[NUM_GROUPS128 * HD128];  // 4 KB
```

每组把 `acc[8]` fp32 cast 为 bf16，用 `float4` 对齐写入（8 bf16 = 16 B = 1 float4）：

```cpp
float4 pack;
__nv_bfloat16* pack_bf = reinterpret_cast<__nv_bfloat16*>(&pack);
#pragma unroll
for (int d = 0; d < 8; ++d) pack_bf[d] = __float2bfloat16(acc[d]);
float4* s_acc_f4 = reinterpret_cast<float4*>(s_acc_bf + gid * HD128);
s_acc_f4[lane] = pack;   // 一次 16-byte 写
```

Group-merge 侧再 `__bfloat162float` 读回做 fp32 加权累加，**fp32 寄存器精度保持不变**，只是中转存储从 fp32 → bf16。

**v2 NCU**：

| 指标 | v1 | v2 | Δ |
|---|---:|---:|---:|
| 单 block SMEM | 24.96 KB | **20.8 KB** | -4 KB |
| **Block Limit SMEM** | 3 | **4** | +1 |
| Theoretical Occupancy | 50 % | **66.67 %** | — |
| Achieved Occupancy | 43.5 % | **56.6 %** | +13 pt |
| DRAM Throughput | 41.9 % | 52.8 % | +11 pt |
| DRAM GB/s | 251 | **316** | +26% |
| **kernel median ms** | **0.0303** | **0.0255** | **1.19×** |

**精度代价**：max |delta| 仍为 4.88e-4，与 v0/v1 完全一致（bf16 中转只在 block-level merge 引入一次 rounding，加权平均 16 项，误差被分子/分母同向吸收）。

---

### v3：探索 + 止损（无收益，2.28× vs 2.31×）

**尝试**：`__launch_bounds__(256, 4)` + 显式 `__fmaf_rn`。

**结果**：
- Registers/thread: 39 → 40（+1），其他指标基本一样
- Duration: 37.50 → 37.92 μs（噪声）
- Median: 0.02548 → 0.02582 ms（噪声）

**结论**：NVCC 在 `-O3` 下早已自动 FMA 化 `acc[d] * exp_scale + p * v_d`；`__launch_bounds__` 对 ptxas 的寄存器分配也没有改变（v2 自然就是 4 blocks/SM 布局）。本轮**没有提升**，被 loop 自动标为 `negative`。

> **教训**：已经吃掉所有"便宜"的结构性优化之后，编译器提示类改动往往零收益。该止损就止损，不要强凑"看起来在优化"的轮次。

**还尝试并放弃过的 v3 候选**：
- `N_SPLIT = 16`：grid → 512 blocks，但每 block 只剩 8 个 micro-tile，pipeline warmup 开销占比变大，median 0.0261 ms → **略逊于 v2**。
- `BN128 = 32`：单 block SMEM 涨到 37 KB，`Block Limit SMEM` 从 4 → 2，Achieved Occupancy 反而下降，median 0.0292 ms → **明显退步**。

---

## 四、NCU 诊断 -> 优化手段映射（本次用到的）

| NCU 信号 | 诊断 | 对应手段 | 本次用到 |
|---|---|---|---|
| `Waves/SM ≪ 1` | Grid 太小，GPU 空载 | 涨 grid：split-K / multi-block-per-row | **v1** |
| `Achieved Occupancy ≪ Theoretical` | Block 内 warp 填不满或 block 少 | 涨 grid 使 SM 满载 | **v1** |
| `Block Limit SMEM = k` 卡住 theoretical | SMEM 是 occupancy 瓶颈 | 降 SMEM：bf16 替代 fp32、warp shuffle 替代 SMEM、分拆 kernel | **v2** |
| `DRAM SOL` 低且 `L2 Hit` 高 | 不是 DRAM-bound，继续找 latency/occupancy | ↑ | v1→v2 |
| `No Eligible > 50%` | Latency-bound，就绪 warp 不足 | 涨 occupancy、`cp.async` 多级流水、ILP | v1（流水已有）、v2（occupancy） |
| `L2 sector util 8/32 B` | 每个 cache sector 利用率低 | MQA/GQA q-head packing；改访存布局 | **未做**（留作下一步） |
| `Mem SOL > Compute SOL` | Memory 侧更忙 | 减访存量或提升访存效率 | 结构已饱和 |

---

## 五、踩过的坑

1. **benchmark harness 不支持 `__nv_bfloat16*`**。skill 自带的 `benchmark.py` 只认 `float*/int*/...`。解决：写 `solve(const float* q, …)` 包装器，把 fp32 randn 输入在**首次调用**一次性转成 bf16 存静态 scratch，benchmark 计时只测真正的 bf16 attention + 一次很小的 bf16→fp32 输出拷贝。

2. **Reference 要匹配 kernel 的数值路径**。PyTorch 用 fp32 做 softmax 会比 bf16 kernel 多 ~1e-2 的误差。解决：reference 里显式把 Q/K/V **先量化到 bf16 再 cast 回 fp32** 做矩阵乘，atol/rtol 设 2e-2。

3. **benchmark buffer size**：`benchmark.py` 默认用最大两维相乘来定 ptr elements。K/V 实际要 `seq_len * num_kv_heads * head_dim = 2097152` elements。必须显式 `--ptr-size=2097152`，否则会 OOB。

4. **NCU 性能计数器需要 root**：机器上 `RmProfilingAdminOnly=1` 使普通用户只能拿到 `LaunchStats` 这类不需要计数器的 section，`--set full` 全部返回空。解决：`sudo -n /usr/local/cuda/bin/ncu …`（本机有免密 sudo for ncu）。写了个 `sudo_ncu.sh` 包装脚本传给 `--ncu-bin` 让 optimize_loop 统一调用。

5. **NCU kernel-name regex 必要**：不过滤时，`ncu --set full` 会把 cublas/pytorch 的 gemv、reduce 这类 warmup kernel 也采样一遍，`targeted_summary.txt` 里第一个出现的是 cublasGemv，不是你的 kernel。用 `--kernel-name-regex="flash_decode_bf16_hdim128_splitk_pass1_kernel"` + `--launch-skip=40 --launch-count=1` 精准抓到自己的 kernel 的稳态一次 invocation。

6. **Loop 的 strategy fingerprint 机制很严格**。`optimization_proposal.md` 里的 `## Strategy tags` 如果不改，loop 会把你这一版标记为 `blocked_strategy_reused`，即使 benchmark 明显变快也不给 positive。需要每轮都填真实的 tag（e.g. `split_k_on_seq`、`s_acc_bf16_demote`）才能正确归类。

7. **Split-K 的 pass-1 输出必须是未归一化的 `O_i`**。常见错误是直接写 `O_i / l_i`，这样 pass-2 无法正确做 log-sum-exp 合并。记录 `(m_i, l_i, 原始 acc_i)` 三元组才行。

8. **回写 RustInfer 时 chunk_size 必须在 device 端计算**。我最初的版本用了 host 端硬编码 `chunk_size = 4096` → 对 seq_len=2048 只会让 split 0 做事，退化回原版本。修正：kernel 里读 `*seq_len_ptr + 1` 后自己算 `chunk_size = ceil(seq_len/N_SPLIT)` 向上取整到 `BN128`。

---

## 六、对 RustInfer 的改动

- 只改了 `crates/infer-core/src/op/kernels/cuda/flash_attn_gqa/flash_decoding_bf16_hdim128.cu` 一个文件。
- `extern "C" void flash_decoding_cu_bf16_hdim128(...)` **签名完全不变**，`mod.rs` 无需改。
- 新 kernel 内部用一个进程全局的 static workspace 缓存 `(M_g, L_g, O_g)`（`cudaMallocAsync`，按 `(num_q_heads, head_dim)` 懒分配一次，大小 O(1) 与 seq_len 无关）。
- `cargo build --release --features cuda -p infer-core` 通过（43.75s，只有 qwen3.rs 里的 `unused_mut` 旧告警，与本改动无关）。

---

## 七、还能继续优化的方向（下一轮可接）

按 NCU v2/v3 剩余信号排序，预期收益从高到低：

1. **MQA/GQA q-head packing**（预期 +30~50%）：当前 `group_size=4` 个共享 KV 的 q-head 会分别在 4 个 block 里重复读同一组 K/V。合并到单 block 后：
   - KV 全局读少 4×（DRAM 压力大降）
   - NCU 报告的 "L2 sector util 8/32 B per sector" 这条规则直接解决
   - 但会重写 grid 布局，SMEM 需要同时装 4 个 q-head 的状态，可能需要从 256 threads/block 涨到 512

2. **Warp specialization**（预期 +10~20%）：8 warps/block 拆成 2 个 producer warp（专跑 `cp.async`）+ 6 个 consumer warp（专跑计算），硬件级生产者-消费者，继续压 `No Eligible` 47%。

3. **Tensor Core MMA for Q·K**（预期 +5~15%）：A10 3rd-gen TC 支持 bf16 MMA。decode `q_seq_len=1` 下直接 MMA 比较尴尬（需要 padding 到 16），但可以用 `mma.m16n8k16` 让 `Q (1×128)` 和 `K (16×128)` 的点积一次出 16 个 score，替掉 `__hmul2 + shuffle` 路径。

4. **`num_stages=3` 流水**（预期 +3~8%）：把 cp.async 双 buffer 扩成三 buffer，`Warp Cycles/Issued` 从 12.8 可能降到 10 左右。

**但注意**：v2 已经到了 DRAM SOL 53%、Achieved Occupancy 57%，再想加速 2× 以上基本必须上 (1)，因为其它都是边际收益。

---

## 八、方法论沉淀（可复用到其他 kernel）

1. **先跑一次 baseline + full NCU**。不要在没有基线指标的情况下做任何优化假设。
2. **Roofline 一级判定**（SM SOL vs Memory SOL）决定后续方向：
   - SM + Memory 双低 → Latency Bound → 涨 occupancy / 涨 grid（本次 v0 就是这里）
   - Memory 高 → 减访存量 / 用 Tensor Core / kernel fusion
   - SM 高 → Tensor Core / 强度削减 / 分支消除
3. **每轮只解 1-2 个最明确的瓶颈**，不要同时改 5 个变量，否则 NCU 不知道哪个改动有效。
4. **提升最大的那一步通常是"把算法对齐硬件拓扑"**（split-K 是把 `num_q_heads × seq_len` 两个独立维度都投影到 grid），而不是指令级优化。
5. **SMEM 是 occupancy 最常见的瓶颈**。先把 `Block Limit Shared Mem` 看清楚，再决定要不要砍 SMEM。
6. **benchmark 的 kernel median 是北极星**；NCU 只是辅助诊断。NCU 报告哪个指标"看起来好"不代表 kernel 真的更快（比如 v3 的 Achieved Occupancy 甚至比 v2 略高，但 median 没降）。
7. **记得止损**。当一轮改动在 benchmark 上 ≤ 1% 差异且 NCU 指标基本持平，说明你已经进入该算子结构的 local optimum，继续下去就该换结构（split-K 类型的宏观改动）或换硬件路径（Tensor Core、warp specialization）。

---

## 九、产物索引

**优化运行目录**（包含所有 4 轮的代码、benchmark、NCU 报告、proposal）：
```
/data/home/vinciiliu/cuda-optimized-skill/flash_decoding_bf16_hdim128_opt/
├── v0_baseline.cu                              # baseline + solve() 包装
├── v1_splitk.cu                                # split-K N_SPLIT=8
├── v2_splitk_smem.cu                           # ★ winner: split-K + bf16 s_acc
├── v3_launchbounds_fma.cu                      # launch_bounds + FMA（没收益）
├── reference.py                                # PyTorch bf16 reference
├── sudo_ncu.sh                                 # ncu wrapper（绕过权限限制）
└── optimize_runs/run_main/
    ├── final_summary.md                        # 自动汇总所有轮次
    ├── run_manifest.json                       # 每轮参数 + strategy memory
    ├── preflight_check.md/.json
    ├── iter_v0/
    │   ├── v0_baseline_v0.cu
    │   ├── benchmark_result.json
    │   ├── targeted.ncu-rep / full.ncu-rep
    │   ├── targeted_summary.txt / full_summary.txt
    │   ├── iteration_summary.md
    │   └── optimization_proposal.md            # v0 瓶颈诊断 + v1 计划
    ├── iter_v1/ ... (同构)
    ├── iter_v2/ ... ★ 最佳版本路径
    └── iter_v3/
```

**已回写的生产代码**：
```
/data/home/vinciiliu/RustInfer/crates/infer-core/src/op/kernels/cuda/flash_attn_gqa/
└── flash_decoding_bf16_hdim128.cu              # 已替换为 split-K + bf16 s_acc 版
```

**最佳版本 NCU 命令（复现用）**：
```bash
sudo -n /usr/local/cuda/bin/ncu \
  --target-processes all --profile-from-start on \
  --launch-skip 40 --launch-count 1 \
  --set full \
  --kernel-name-base demangled -k regex:flash_decode_bf16_hdim128_splitk_pass1_kernel \
  -o v2_full -f \
  python benchmark.py v2_splitk_smem.cu \
    --backend=cuda --ref=reference.py \
    --num_q_heads=32 --num_kv_heads=8 --head_dim=128 --seq_len=2048 \
    --ptr-size=2097152 --warmup=20 --repeat=50 --arch=sm_86
```

---

## 十、一句话总结

**原版单 kernel 单 head-per-block 的 flash decoding 在 72-SM 的 A10 上只用了 0.15 waves/SM，绝大部分 SM 空转；引入 KV 方向的 split-K 把 grid 从 32 涨到 256 解决主矛盾（1.94×），再把 block 内 merge 用的 fp32 scratch 降 bf16 解开 SMEM 卡住的 occupancy（累计 2.31×）。剩下的 compiler-hint 类小优化已进入边际收益区，再大幅提升需要结构性改动（GQA q-head packing / Tensor Core）。**
