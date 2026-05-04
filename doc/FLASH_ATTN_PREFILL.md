# Flash-Attention Prefill Kernel：从零开始的学习文档

> 目标读者：假设你 **完全没写过 CUDA、也没读过 FlashAttention 论文**。看完这篇你应该能：
> 1. 说清楚 attention 在干什么；
> 2. 说清楚 FlashAttention 为什么比朴素写法快；
> 3. 看懂我们这份 kernel 的每一个关键片段在干什么；
> 4. 知道想改/扩它该从哪里下手。

对应代码：
- Kernel 主体：`crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_attn_gqa_prefill.cu`
- Rust 封装：`crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/mod.rs` 里 `flash_attn_gqa_prefill`
- 上层 op：`crates/infer-worker/src/op/flash_gqa.rs` 里 `FlashAttnGQA::forward`
- 测试：`test_flash_attn_gqa_prefill_cuda_vs_cpu`（同文件）

---

## 0. 术语字典（先背下来，后面反复用）

| 名字 | 含义 |
|---|---|
| **Q / K / V** | Query / Key / Value，三个形状一样的大矩阵 |
| **seq_len / q_len / kv_len** | 序列长度，就是"这句话有多少个 token" |
| **head_dim (HD)** | 每个 attention head 的向量维度，常见 64 / 128 |
| **num_q_heads (Hq)** | Query 的 head 个数 |
| **num_kv_heads (Hkv)** | Key/Value 的 head 个数。**GQA** 就是 Hq > Hkv |
| **batch (B)** | 一次同时处理几条序列 |
| **prefill** | 一次性吃完 prompt（q_len 很大，比如 512） |
| **decode** | 每次只算 1 个新 token（q_len = 1） |
| **causal** | 只能看"自己和之前"，预测未来不许作弊 |
| **MMA** | 张量核心矩阵乘指令（`mma.sync.aligned...`） |
| **smem** | shared memory，SM 上的小块高速内存（~100KB） |
| **gmem** | global memory，显存，大但慢 |
| **cp.async** | Ampere 引入的异步 gmem→smem 搬运指令 |
| **ldmatrix** | 从 smem 加载到寄存器、自带 warp 内 shuffle，给 MMA 喂数据 |
| **tile** | 把大矩阵切成小方块，每次算一块 |

---

## 1. Attention 到底在算什么

一行公式就是全部：

$$
O = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

当 `Q` 形状是 `[q_len, d]`，`K`、`V` 形状是 `[kv_len, d]`：

1. `S = Q @ K^T` → `[q_len, kv_len]`，叫 **scores**。
2. `S /= sqrt(d)`（放缩，防止 softmax 爆炸）。
3. 对 S 的**每一行**做 softmax → `P`，形状不变。
4. `O = P @ V` → `[q_len, d]`。

**直觉**：每个 Q 行（一个 token 的 query 向量）跟每个 K 行（另一个 token 的 key 向量）做点积，得到"我跟你有多相关"的分数；softmax 归一化后作为权重，对所有 V 行做加权平均，就是"我该看看什么"。

**GQA**：Hq 个 Q head 共享 Hkv 个 KV head，`kv_head_idx = q_head_idx / (Hq / Hkv)`。节省 KV cache 显存。

**Causal**：LLM 的训练/prefill 里，位置 `i` 的 Q 只能看位置 `j ≤ i + (kv_len - q_len)` 的 K/V。超出的位置把 score 设成 `-inf`。

---

## 2. 朴素实现为什么慢

把上面四步直接写出来：

```
for 每个 head:
  S = Q @ K^T       # [q_len, kv_len] 的矩阵，存在显存里
  S /= sqrt(d)
  P = softmax(S)    # 再扫一遍，又写一遍 S
  O = P @ V
```

**致命问题**：`S` 这个矩阵本身可能超大。
- q_len = kv_len = 8192 时，`S` 就是 `8192 × 8192 × 4 bytes = 256 MB`。
- 不仅占显存，关键是要 **写一次显存 + 读一次显存** 才能做 softmax，再读一次才能乘 V。**显存带宽成了瓶颈**。

---

## 3. FlashAttention 的核心思想：online softmax + tiling

FlashAttention（Tri Dao, 2022）解决方案：

> **永远不要把完整的 S 写到显存。把 Q/K/V 分块，在 shared memory 里算 S 的一小块，立刻消费掉，只把最终 O 写出去。**

但 softmax 是全局归一化（要知道整行最大值和总和），怎么能"只看一小块就消费掉"？

### 3.1 Online softmax（最关键的数学）

设某一行我们分 T 个 tile 处理。每处理完一个 tile，维护三样东西：

- `m`：当前见过的最大 score
- `l`：当前 `sum(exp(score - m))`
- `o`：当前"加权累加的 V"

来新 tile 时，它自己有 `(m_new, l_new, o_new)`。**合并规则**：

$$
\begin{aligned}
m_{out} &= \max(m, m_{new}) \\
l_{out} &= e^{m - m_{out}} \cdot l + e^{m_{new} - m_{out}} \cdot l_{new} \\
o_{out} &= e^{m - m_{out}} \cdot o + e^{m_{new} - m_{out}} \cdot o_{new}
\end{aligned}
$$

最后 `O_final = o_out / l_out`。

这个等式保证了：**即使分块处理，结果也跟一次性做 softmax 完全一样**（数学上恒等）。

### 3.2 Tiling（切块）

我们选 `BlockM = 128` 行 Q 和 `BlockN = 64` 行 K/V 作为一块。然后：

```
for 每个 Q-block (128 行):
    load Q-block 到 smem
    初始化 m=-inf, l=0, o=0  (都在寄存器)
    for 每个 KV-block (64 行):
        load K-block、V-block 到 smem（用 cp.async 重叠）
        S = Q-block @ K-block^T         # [128, 64] 在寄存器
        mask + scale
        m_new, l_new, p = softmax(S)     # online
        o += e^{m-m_out} * o(旧) + p @ V-block  # 合并
        m = m_out; l = l_out
    O[Q-block] = o / l                   # 一次性写回
```

**这就是 FlashAttention prefill 的全部骨架**。我们的 kernel 就是这个骨架的 SM80 高性能实现。

---

## 4. GPU 执行模型快速补课

你要看懂 CUDA kernel 就得知道这几层：

```
Grid    (gridDim)        多个 Block
  └─ Block (blockDim)    多个 Warp（每 32 thread 一组）
        └─ Warp         32 个 Thread
              └─ Thread  最小执行单位
```

- **每个 Block 独占一份 shared memory**（几十到上百 KB）。
- **每个 Thread 拥有寄存器**（很快但很少）。
- **Warp** 内 32 个 thread 锁步执行，可以高效 shuffle。
- **Tensor Core**：硬件矩阵乘单元，用 `mma.sync` 一条指令算一个 `16×8×16` 的矩阵乘。

我们的 kernel 线程布局：
- `blockDim = (32 * 4) = 128` 个 thread（= **4 个 warp**）
- 每个 block 负责**一个 (batch, q_head, Q-tile)** 的全部工作
- `gridDim = (m_blocks, num_q_heads, batch)`，`m_blocks = ceil(q_len / 128)`

---

## 5. 我们 kernel 的关键设计（逐条对照代码）

打开 `flash_attn_gqa_prefill.cu`，结合下面一起看。

### 5.1 Traits：把所有静态配置打包

```cpp
template <class Elem_, int HeadDim_>
struct KTraits {
    using Elem = Elem_;
    static constexpr int HeadDim = HeadDim_;
    using SmemAtom = decltype(composition(
        Swizzle<3,3,3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using SmemLayoutQ  = decltype(tile_to_shape(SmemAtom{},
                                    Shape<Int<kBlockM>, Int<HeadDim>>{}));
    ...
    using Mma = decltype(make_tiled_mma(
        typename MmaAtomFor<Elem>::type{},
        Layout<Shape<Int<kNWarps>, _1, _1>>{},   // 4 warps 沿 M 方向
        Tile<Int<16*kNWarps>, _64, _64>{}));     // MMA tile (64, 64, 64)
    ...
};
```

**一句话**：`KTraits<Elem, HeadDim>` 就是"这一组 dtype + head_dim 下所有 smem layout、MMA、copy atom 的静态描述"。编译器拿到它以后，所有东西都是编译期常量，可以极致优化。

> **Swizzle<3,3,3>**：smem 地址的一种异或映射。目的是让 `ldmatrix` 在加载时不会 bank conflict。你不理解细节也没关系，**记住"写 smem 和读 smem 的地址是经过 Swizzle 的，所以不能当成普通行列"** 就够了。

### 5.2 全局 tensor：带真实 stride

```cpp
auto Q = make_tensor(make_gmem_ptr(q_bh),
                     make_shape(q_len, Int<HD>{}),
                     make_stride(q_stride_s, _1{}));
```

这里最重要：**stride 是 kernel 参数传进来的，不是硬编码**。
- 传入 NHD 连续布局：`q_stride_s = Hq * HD`，一切正常
- 传入非连续 view：`q_stride_s = 某个奇怪的值`，kernel 也能正确寻址

这就是 "stride-aware"。传统写法会假设 `addr = row * (Hq*HD) + head*HD + d`，一旦 tensor 是 slice 或者 HND 布局就崩；我们把 stride 交给 cute，cute 负责算地址。

对比 flashinfer 的 `batch_attention.cu` 第 85-103 行，它也在干同样的事：从 PyTorch tensor 取 stride，传给 kernel。工业级实现的共同必备能力。

### 5.3 Block 负责的工作区域

```cpp
const int block_m    = blockIdx.x;
const int q_head_idx = blockIdx.y;
const int batch_idx  = blockIdx.z;
const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

const Elem* q_bh = q_ptr + batch_idx * q_stride_b + q_head_idx  * q_stride_h;
const Elem* k_bh = k_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h;
...
```

每个 block 先按 `(batch, head)` 给自己选好起点。注意 **K/V 的 head 是 `kv_head_idx`**（GQA 除法），这就是 GQA 的精髓——Hq 个 Q head 里每 `groups = Hq/Hkv` 个共用一个 KV head。

之后 `gQ = local_tile(Q, Shape<128, HD>, make_coord(block_m, 0))` 切出自己负责的 Q tile。

### 5.4 cp.async：让 GPU 一边搬一边算

朴素加载：
```cpp
smem[i] = gmem[addr];   // 这条指令会 stall 直到数据到
```

cp.async：
```cpp
cp.async(smem, gmem);   // 发起搬运请求，立刻返回
// ... 这里随便干别的事情 ...
cp_async_wait<0>();     // 在真正需要数据前再等
```

**好处**：搬运和计算可以同时进行，把显存延迟"藏"到计算后面。

我们 kernel 的流水线：

```
[初始化]
  load Q 整块 → smem         cp_async_fence();   // group 0
  load K[0]    → smem         cp_async_fence();   // group 1

[主循环 for nt = 0..N-1]
  load V[nt]   → smem         cp_async_fence();   // group 2
  wait<1>                     // 等到 K[nt] 就绪（V 还在飞）
  ┌─ S = Q @ K[nt]^T          // tensor core
  ├─ load K[nt+1] → smem      cp_async_fence();   // 同时发起下一块 K 预取
  ├─ mask + online softmax
  ├─ wait<1>                  // 等到 V[nt] 就绪
  └─ O += P @ V[nt]
```

**关键技巧**：wait<N> 表示"允许 N 个最新的 group 还没完成"。我们始终让下一个 K 的预取处于 in-flight 状态，这样 compute 就被 latency-hidden 了。

### 5.5 MMA：Tensor Core 矩阵乘

```cpp
auto rQ = thr_mma.partition_fragment_A(sQ);
auto rK = thr_mma.partition_fragment_B(sK);
auto rS = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

cute::copy(s2r_q, tXsQ, tXrQ);   // ldmatrix smem→reg
cute::copy(s2r_k, tXsK, tXrK);
cute::gemm(mma, rQ, rK, rS);     // mma.sync 128×64×HD
```

- `partition_fragment_*` 帮每个 thread 分到它应该持有的那一小撮 tensor core 输入/输出寄存器
- `ldmatrix` 是一条特殊 warp 级指令，一次拉 4 个 `8×8` 矩阵到 32 个 thread 的寄存器，并自动按 MMA 需要的形状摆好
- `cute::gemm` 在 M/K 维展开多个 `mma.sync`

**你不需要手写 mma PTX，cute 都包好了**。唯一要注意：M, N, K 方向的 tile 大小必须跟 MMA atom 对齐。

### 5.6 Online softmax 实现

```cpp
if (nt == 0) {
    reduce_max_rows<true>(scores, row_max);
    scale_apply_exp2(scores, row_max, scale_log2);
    reduce_sum_rows<true>(scores, row_sum);
} else {
    Tensor m_prev = make_fragment_like(row_max);
    cute::copy(row_max, m_prev);
    reduce_max_rows<false>(scores, row_max);   // m_out = max(m_prev, m_new)
    // 重标定老的 o 和 l
    for mi: sc = exp2((m_prev - m_out) * log2e_scale)
            row_sum[mi] *= sc
            rO[mi][:]   *= sc
    scale_apply_exp2(scores, row_max, scale_log2);   // P = exp2(S - m_out)
    reduce_sum_rows<false>(scores, row_sum);          // row_sum += sum(P)
}
...
gemm_rs(rO, tOrP, rVt, tOsVt, mma, ...);   // rO += P @ V
```

对照 §3.1 的公式：每来一个新 tile，就按 `exp(m_prev - m_out)` 重标定之前累积的 `o` 和 `l`，然后加上新 tile 的贡献。正确性就是那个恒等式保证的。

**实现细节**：
- 用 `exp2` 而不是 `exp`。因为 `exp2f` 在 GPU 上是 native 指令；我们提前把 `softmax_scale * log2(e)` 吸到 scale 里。
- `row_max / row_sum` 是寄存器 tensor，**每个 thread 只持有自己 MMA 片段的行** —— 所以需要 `quad_allreduce` 在 warp 的 "quad"（4 连通组）内做 reduce，这对应 MMA-C 的行布局。

### 5.7 Mask（causal + 边界）

```cpp
if (is_causal) {
    apply_causal_mask(rS, kBlockN*nt, row_idx_ofs, 16*kNWarps,
                      kv_len - q_len, kv_len);
} else {
    if ((nt + 1) * kBlockN > kv_len) {
        apply_col_bound_mask(rS, kBlockN*nt, kv_len);
    }
}
```

两种 mask 都是把 `rS` 里越界位置设成 `-INFINITY`，这样 `exp2(-inf) = 0`，对 softmax 不贡献。

- `causal_shift = kv_len - q_len`：Q 的第 `i` 行（block 内）的"绝对位置"是 `current_kv_len + i`，它能看到的最大 K 列是 `row + causal_shift`
- 非 causal 只需要在 **最后一个** KV tile 做列边界 mask（前面的 tile 肯定都落在 `kv_len` 内）

### 5.8 Ragged 的 predicate load

q_len 不是 128 整数倍、kv_len 不是 64 整数倍时怎么办？

```cpp
// 加载 Q：按行号 predicate
for m in 0..size<1>(tQgQ):
    row = block_m * kBlockM + get<0>(tQcQ(0, m, 0));
    if row < q_len:
        cute::copy(copy_q, tQgQ(_, m, _), tQsQ(_, m, _));
    else:
        cute::clear(tQsQ(_, m, _));     // 越界行填 0，防止脏数据污染 S
```

K/V 同理。写回 O 的时候也再做一次行越界检查。**关键原则**：**越界位置永远是 0（for 数据）或 -inf（for score）**，这样前向计算路径不需要分支，性能不受影响。

### 5.9 Head_dim 分发

```cpp
switch (head_dim) {
case 64:  return launch_impl<Elem,  64>(...);
case 128: return launch_impl<Elem, 128>(...);
case 192: return launch_impl<Elem, 192>(...);
case 256: return launch_impl<Elem, 256>(...);
default:  return cudaErrorInvalidValue;
}
```

**为什么不支持任意 head_dim？** 因为 smem layout、MMA tile 大小都是**编译期常量**，不同 head_dim 会编译出不同的 kernel 二进制。想扩一个 head_dim=96？加一行 case 就行（但 96 不是 64 倍数，smem atom 需要调整，所以当前不支持）。

**`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)`**：head_dim=256 时 smem 要 128KB，超过默认 48KB 限制，必须 opt-in。

---

## 6. Rust 侧是怎么调的

### 6.1 FFI 声明（`kernels/cuda/flash_attn_gqa/mod.rs`）

```rust
unsafe extern "C" {
    pub fn launch_flash_attn_prefill_bf16(
        q: *const half::bf16, qsb: i64, qss: i64, qsh: i64,
        k: *const half::bf16, ksb: i64, kss: i64, ksh: i64,
        v: *const half::bf16, vsb: i64, vss: i64, vsh: i64,
        o: *mut   half::bf16, osb: i64, oss: i64, osh: i64,
        batch: i32, q_len: i32, kv_len: i32,
        num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
        softmax_scale: f32, is_causal: i32,
        stream: cudaStream_t,
    );
    // ... fp16 同款
}
```

这 12 个 `i64` 就是 4 个张量 × 3 个 stride（batch, seq, head），加上 head_dim 维 stride 隐含为 1。

### 6.2 Safe wrapper（同文件）

```rust
pub unsafe fn flash_attn_gqa_prefill(
    q, k, v, o: &Tensor,
    q_seq_len, current_kv_len_host,
    num_q_heads, num_kv_heads, head_dim,
    is_causal, cuda_config,
) -> Result<()> {
    // 1. 从 Tensor 抽 stride
    let qss = q.strides()[0] as i64;   // 沿 seq 的 stride，用 Tensor 的真实 stride
    let qsh = head_dim as i64;         // 沿 head 的 stride = head_dim
    ...
    let kv_len_total = current_kv_len_host + q_seq_len;  // past + new
    // 2. softmax scale = 1/sqrt(head_dim)
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    // 3. dtype 分发
    match dtype {
        BF16 => launch_flash_attn_prefill_bf16(...),
        F16  => launch_flash_attn_prefill_fp16(...),
    }
}
```

**这里的重点**：Rust 侧负责"对 Tensor 语义（shape/stride/dtype）做翻译"；CUDA kernel 只接收数字和指针，不关心 PyTorch-like 的抽象。

### 6.3 上层 op（`op/flash_gqa.rs`）

```rust
DeviceType::Cuda(_) => {
    let kv_len_host = ... // CPU tensor 直读；GPU tensor 做一次 to_cpu
    if q_seq_len == 1 {
        // decode 路径：走老 split-K kernel（需要 device 指针）
        kernels::cuda::flash_attn_gqa(...)
    } else {
        // prefill 路径：走新 stride-aware kernel
        kernels::cuda::flash_attn_gqa_prefill(
            input_q, input_k_cache, input_v_cache, output_o,
            q_seq_len, kv_len_host,
            num_q_heads, num_kv_heads, head_dim,
            causal, cuda_config,
        )?
    }
}
```

**q_seq_len == 1 走 decode（flash decoding，split-K），其他走 prefill**。这是 LLM 推理框架的标准路由。

---

## 7. 测试怎么写的、怎么读

测试文件：`op/flash_gqa.rs` 里的 `test_flash_attn_gqa_prefill_cuda_vs_cpu`。

思路很直：

```
for dtype in [BF16, FP16]:
  for (Hq, Hkv, HD, q_len, past, causal) in cases:
    1. 随机生成 f32 数据
    2. 转成对应 dtype，放到 CPU Tensor 里
    3. 用 CPU reference 算一次（FlashAttnGQA::forward on CPU）→ o_cpu
    4. 同样数据放到 GPU，CUDA 路径算一次 → o_gpu
    5. 都转成 f32 比对，逐元素 |a-b| <= atol + rtol*|a|
```

测试矩阵覆盖：
- dtype × head_dim × causal × GQA 比例 × **non-aligned q_len** × **past_kv > 0**（增量 prefill）

容差：`bf16 atol=6e-2, fp16 atol=1e-2`。实测误差 ~3.9e-3（bf16 mantissa 精度极限），完全 OK。

> **为什么不严格对齐？** 因为 CPU ref 是 2-pass exact softmax，GPU 是 online softmax + FP32 累加路径，**中间 rounding 不可能字节相同**。只要每个元素落在 bf16 精度误差范围内就算对。

运行：

```bash
cd /root/RustInfer
CUDA_ARCH=sm_80 cargo test --features cuda -p infer-worker --lib \
    op::flash_gqa::tests::test_flash_attn_gqa_prefill_cuda_vs_cpu \
    --release -- --nocapture
```

输出形如：
```
prefill cuda-vs-cpu  dtype=BF16 Hq=8 Hkv=2 HD=64 Qn=128 past=0 causal=true max_abs=3.9e-3 bad=0/65536
...
test result: ok. 1 passed; 0 failed
```

`bad` 是超出容差的元素数，必须是 0。

---

## 8. 怎么加新功能（把这份代码当玩具）

### 8.1 支持新的 head_dim

1. 在 `launch_dispatch` 加一行 `case 96:` / `case 320:`；
2. 确认 smem 够用（`sizeof(SharedStorage<...>) < 共享内存上限`，H100 最多 228KB）；
3. 跑测试；
4. 对非 64 倍数的 head_dim，`SmemAtom` 的 `Shape<_8, _64>` 可能要改成 `_8, head_dim`，swizzle 参数也要重算 —— 当前不支持。

### 8.2 加 batch 支持 ragged（varlen）

当前是 "等长 batch"。要支持每条请求 q_len / kv_len 不同（cu_seqlens 风格）：

1. 加参数 `cu_q_lens[B+1]` 和 `cu_kv_lens[B+1]`（device 指针）；
2. grid.x 改成 "总 m_blocks 数的前缀和"，kernel 里二分查找 `bid`；
3. 每个 block 用 per-request 的 q_len / kv_len 计算 `n_block_max` 和 `causal_shift`；
4. 其他逻辑不动。

参考 FlashAttention 官方 varlen 实现 / flashinfer `batch_attention.cu` 的 plan。

### 8.3 加 paged KV（推理引擎常用）

KV 不连续，存在很多 `[page_size, H, D]` 的 page 里，通过 `page_table` 索引。当前 kernel 需要改：

1. KV 加载的地址从
   `k_base + token_idx * stride_s`
   改成
   `k_base + page_table[page_id] * stride_page + entry_idx * stride_s`；
2. `entry_idx = token_idx % page_size`，`page_id = kv_indptr[req] + token_idx / page_size`。

建议做成独立 kernel 变体（`flash_attn_prefill_paged.cu`），不要和 contiguous 版耦合。

### 8.4 上 Hopper（SM90）

当前是 SM80 特化。想跑到 H100 峰值：

- `cp.async` → `cp.async.bulk` + TMA descriptor
- `SM80_16x8x16` MMA → `SM90_64xNx16` WGMMA
- 加 warp specialization（producer / consumer）
- 加 cluster（thread block cluster）
- 用 DSMEM 做 cluster 内共享

这不是小修，基本上是重写 kernel，可以参考 FA3 / CUTLASS 3.x hopper gemm。

---

## 9. 调试 checklist

写 CUDA kernel 出错时的标准操作：

1. **正确性**：先跑 `cargo test --release ... -- --nocapture` 看 bad 数、max_abs；
2. **illegal memory access / segfault**：
   - 检查 predicate load 是否写越界
   - 检查 stride 是否带单位错（元素 vs 字节）
3. **结果全 0**：
   - 通常是 smem 没对齐，或者 mma 的 K 维度没迭代到（`partition_fragment_A` 形状错）
   - 检查 `extern __shared__` 和 `SharedStorage` 是否 reinterpret 对
4. **部分位置对、部分位置错**：
   - 多半是 mask 逻辑错，打印 `row_idx` / `col_idx` 用单线程 case 确认
5. **NaN**：
   - `exp2f` 输入太大？检查 scale 是不是把 log2e 重复乘了
   - softmax 维护过程中 `m` 变 `inf` 了？检查全 `-inf` 行的保护分支

---

## 10. 推荐阅读顺序

看懂这份 kernel 后继续深入：

1. **FlashAttention-2 paper** (Tri Dao, 2023) —— 读里面的 algorithm 1
2. **CUTLASS CuTe 文档** —— `tile_to_shape`、`partition_fragment_*`、swizzle 讲得最清楚的地方就在官方文档
3. **flashinfer `batch_attention.cu` + `persistent.cuh`** —— 工业级 batched 实现；跟我们的 kernel 相同思路但更复杂
4. **FlashAttention-3 paper** (2024) —— Hopper TMA + WGMMA + warp specialization，是 SM90 的参考
5. **CUDA C++ Programming Guide `cp.async` / `mma.sync` 章节** —— PTX 层面

---

## 11. TL;DR（实在看不下去就只记这段）

- Attention = `softmax(QK^T/√d) V`；朴素实现慢是因为写显存的 S 太大。
- FlashAttention = **分块 + online softmax + 只在 smem 算 S**，显存带宽不再是瓶颈。
- 我们的 kernel：
  - 每个 CUDA block 负责一个 **(batch, q_head, Q-tile)**；
  - BlockM=128 / BlockN=64 / HeadDim∈{64,128,192,256} 静态模板化；
  - Tensor Core 跑 GEMM，cp.async 流水线隐藏搬运延迟；
  - 所有 Q/K/V/O 的 batch/seq/head **stride 都由调用方传入**，任意 view 都能跑；
  - predicate load 处理非对齐的 q_len / kv_len；
  - 只支持 BF16 / FP16 prefill。Decode（q_seq_len=1）走另一个 split-K kernel。
- 正确性靠 `test_flash_attn_gqa_prefill_cuda_vs_cpu` 保证：16 个组合，每个 `bad=0`，误差 ~bf16 量级。
- 想改就改 `KTraits` + `launch_dispatch`，其他基本不用动。

学完这些，你就从"完全不知道 flash attention 是什么"升到"能 review / 扩展这个 kernel"了。继续加油。
