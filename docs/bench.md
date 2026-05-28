# RustInfer 性能基准与诊断手册

本文档记录 RustInfer 当前的性能数据、复现命令、profiling 工具，以及瓶颈
分析方法论。所有数据基于 **H20 GPU + CUDA 13.1 + bf16** 单卡。

---

## 1. 当前基线

### Llama-3.2-1B-Instruct，batch=1，1000 tokens decode

| 模式  | tok/s | µs/tok | 备注 |
|------|------:|-------:|------|
| graph | 685 | 1444 | `step_batch_with_graph` 默认路径 |
| eager | 674 | 1468 | `RUSTINFER_DISABLE_GRAPH=1` |

**Graph 净收益**: 24 µs/tok (1.7%) —— 与理论值 (181 kernel × ~150ns
launch = 27 µs) 吻合。`launch overhead` 占总时间不到 2%，**优化空间在
GPU kernel 内部，不在调度路径**。

### GPU 时间拆解（来自 cudaEvent + nsys）

```
wall:        1444 µs/tok  (685 tok/s)
gpu_graph:   1413 µs/tok  ← cudaGraphLaunch 实测
host:          25 µs/tok  (build_plan + D2H + bookkeeping)

kernel sum:  1130 µs/tok  ← nsys eager 模式的 kernel exec 时间
graph node-edge overhead: 280 µs/tok  ← graph 内部 dependency 串行化
```

---

## 2. 复现命令

### 2.1 Build

```bash
cd /root/RustInfer
cargo build --features="cuda demo" --bin llama3_demo --release
```

输出: `target/release/llama3_demo`

### 2.2 单 prompt benchmark（demo bin）

最常用的快速 A/B 工具。绕过 ZMQ/scheduler，直接 in-process 跑 prefill +
decode：

```bash
./target/release/llama3_demo \
  --model-path /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --prompt "Once upon a time" \
  --max-new-tokens 1000 \
  --max-seq-len 2048
```

输出:
```
[demo] generated 804 tokens in 1.17s (685.9 tok/s)
```

### 2.3 GPU profiling（cudaEvent 打点）

`RUSTINFER_PROFILE_GPU=1` 在 `step_batch_with_graph` 包一对 `cudaEvent`，
分别累计：
- `wall_t0..elapsed`：CPU 端从入口到出口的 wall time
- `cudaEventElapsedTime(t0,t1)` 围 `cudaGraphLaunch`：纯 GPU 时间

```bash
RUSTINFER_PROFILE_GPU=1 ./target/release/llama3_demo \
  --model-path /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --prompt "Once upon a time" --max-new-tokens 1000 --max-seq-len 2048 \
  2>&1 | grep -E "tok/s|profile"
```

输出:
```
[demo] generated 804 tokens in 1.17s (685.9 tok/s)
[profile] decode steps=804  wall=1444.2µs/tok  gpu_graph=1413.6µs/tok  \
          host_overhead=25.0µs/tok (1.7%)
[profile] tok/s ceiling if host_overhead → 0: 707.4
```

### 2.4 Eager vs Graph A/B

```bash
echo "=== GRAPH ==="
RUSTINFER_PROFILE_GPU=1 ./target/release/llama3_demo \
  --model-path /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --prompt "Once upon a time" --max-new-tokens 1000 --max-seq-len 2048 \
  2>&1 | grep -E "tok/s|profile"

echo "=== EAGER ==="
RUSTINFER_PROFILE_GPU=1 RUSTINFER_DISABLE_GRAPH=1 \
  ./target/release/llama3_demo \
  --model-path /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --prompt "Once upon a time" --max-new-tokens 1000 --max-seq-len 2048 \
  2>&1 | grep -E "tok/s|profile"
```

### 2.5 nsys kernel 占比

```bash
rm -f /tmp/llama_profile.*
RUSTINFER_DISABLE_GRAPH=1 nsys profile -t cuda --stats=true \
  --force-overwrite=true -o /tmp/llama_profile \
  ./target/release/llama3_demo \
    --model-path /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
    --prompt "Once upon a time" --max-new-tokens 100 --max-seq-len 512 \
  2>&1 | grep -A 22 cuda_gpu_kern_sum | head -22
```

**重要约束**：CUDA Graph 模式下 nsys 看不到 graph 内部 kernel；要测
kernel 级 breakdown 必须 `RUSTINFER_DISABLE_GRAPH=1`。

### 2.6 多 batch / 在线场景（end-to-end via HTTP server）

走完整链路（infer-server + scheduler + worker），用 Python harness 并发
打 HTTP：

```bash
# 启动 server（后台跑）
PATH=$PWD/target/release:$PATH ./target/release/rustinfer-server \
  --model /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --model-type llama3 --device cuda:0 \
  --max-batch-tokens 8192 --max-batch-seqs 64 --max-model-len 4096 \
  --port 8000 &

# 跑 throughput sweep
python3 bench/bench_batch_throughput.py \
  --label llama3-1b-h20 \
  --max-tokens 128 \
  --batches 1,2,4,8,16,32
```

数据集: `bench/bench_prompts.json` (51,906 条 Alpaca-style prompts)。

其他工具:
- `bench/bench_ttft_tpot.py`：测 TTFT (time-to-first-token) 和 TPOT
  (time-per-output-token) 分布
- `bench/bench_real_arrival.py`：泊松到达模拟，测延迟尾部
- `bench/bench_online.py`：长跑稳定性 + 内存泄漏

---

## 3. 环境变量参考

| 变量 | 作用 | 用途 |
|------|------|------|
| `RUSTINFER_DISABLE_GRAPH=1` | 关闭 CUDA Graph，走 eager 路径 | A/B graph vs eager |
| `RUSTINFER_PROFILE_GPU=1` | 打 cudaEvent 量 GPU vs host 时间 | 量化调度延迟 |
| `RUSTINFER_TRACE_GRAPH=1` | 每次 graph replay 打印 slot | 验证 graph 真的在跑 |
| `RUSTINFER_FORCE_GEMM=1` | bf16 m=1 强制走 cuBLASLt（不用 hgemv）| GEMV vs GEMM A/B |
| `RUSTINFER_DEBUG_LAYERS=1` | 每层打印中间张量首 8 个值 | 数值正确性诊断 |
| `RUSTINFER_TEST_TOKENIZER` | 指向 tokenizer.json 路径 | infer-server 单测 |

---

## 4. Kernel 数量账本（Llama-3.2-1B forward）

每层 11 个 kernel：

```
 1. qkv_proj             hgemv_bf16_v3
 2. rope                 rope_rotate_kernel
 3. scatter_kv           scatter_kv_paged_kernel
 4. attention pass1      paged_decode_pass1_kernel
 5. attention combine    paged_decode_combine_kernel
 6. o_proj               hgemv_bf16_v3
 7. fused_add_rmsnorm    (residual + post_attn_norm)
 8. gate_up_proj         hgemv_bf16_v3
 9. swiglu_packed        swiglu_packed_kernel
10. down_proj            hgemv_bf16_v3
11. fused_add_rmsnorm    (residual + 下一层 input_norm)
```

**16 layers × 11 = 176**

头尾 5 个：

```
embedding + layer 0 input_layernorm + lm_head (cuBLAS) + argmax_phase1 + argmax_phase2 = 5
```

**Total ≈ 181 kernel/forward**

---

## 5. 每 forward 字节预算（带宽下限分析）

权重读取（bf16）:

| Op | shape | bytes/forward | 16 layers |
|---|---|---:|---:|
| qkv_proj | [6144, 2048] | 25 MB | 400 MB |
| o_proj | [2048, 2048] | 8 MB | 134 MB |
| gate_up_proj | [16384, 2048] | 67 MB | 1073 MB |
| down_proj | [2048, 8192] | 33 MB | 537 MB |
| **layer total** | | **133 MB** | **2144 MB** |
| lm_head | [128256, 2048] | 525 MB | — |

**总权重读取**: ~2.65 GB / forward

**H20 显存带宽**: 4000 GB/s (HBM3, 实测 ~3500 GB/s)

**理论下限**:
```
2.65 GB / 4000 GB/s = 0.66 ms/tok  (1515 tok/s, 100% bw)
2.65 GB / 3500 GB/s = 0.76 ms/tok  (1316 tok/s, 87% bw)
```

实测 nsys kernel sum = 1.13 ms/tok → **当前带宽利用率约 67%**。

---

## 6. 性能演进

| 版本 | tok/s | 关键改动 |
|------|------:|---------|
| 起点（commit 271e126 后） | 415 | 基线，paged KV 全链路打通 |
| ForwardWorkspace | 555 | 消除 per-step `cudaMalloc` |
| BatchWorkspace + upload_async | 600 | plan H2D 异步化 |
| CUDA Graph (Phase 5f) | 622 | host launch 折叠 |
| lm_head 走 cuBLASLt（N>16384 阈值）| 637 | -3% lm_head GEMV 时间 |
| Fused gate_up GEMV | 660 | 每层省 1 个 GEMV launch |
| Zero-copy QKV split via narrow | 689 | 每层省 3 个 split_cols launch |

**当前**: 685-690 tok/s

---

## 7. 瓶颈分析方法论

按因果链定位：

```
wall (1444 µs)
  ├─ host_overhead (25 µs, 1.7%)
  │   ├─ build_plan (Vec clone + 9× upload_async)
  │   └─ to_host_vec (D2H + sync)
  └─ gpu_graph (1413 µs, 98.3%)
      ├─ kernel exec sum (1130 µs)  ← nsys 测
      │   ├─ hgemv (770 µs, 55%)
      │   ├─ flash_decode pass1+combine (160 µs)
      │   ├─ fused_add_rmsnorm (95 µs)
      │   ├─ scatter_kv (55 µs)
      │   ├─ swiglu_packed (50 µs)
      │   ├─ rope (45 µs)
      │   └─ lm_head + argmax (155 µs)
      └─ graph barrier overhead (280 µs)
```

**诊断命令对应**:
- `host_overhead` → `RUSTINFER_PROFILE_GPU=1`
- `gpu_graph` → `cudaEventElapsedTime`（已打点）
- `kernel exec sum` → `nsys ... RUSTINFER_DISABLE_GRAPH=1`
- `graph barrier overhead` → `gpu_graph - kernel sum`

---

## 8. 当前最高 ROI 优化方向

要破 800 tok/s（每 token 1.25 ms），按数据指引：

| # | 优化 | 预估节省 | 难度 | 备注 |
|--:|------|---------:|------|------|
| 1 | hgemv kernel 升级（cp.async 双 stage + sm_90 wgmma）| 200 µs | 高 | 目标带宽利用率 67% → 85% |
| 2 | fuse paged_decode pass1 + combine | 30 µs | 中 | 当前 split=8 时 reduce 步骤可内联 |
| 3 | fuse rope + scatter_kv | 30 µs | 中 | 一次内核同时写 RoPE 和 paged 槽 |
| 4 | reduce graph node 数（fuse #2 + #3 后 11 → 9 kernel/层） | 50 µs | 中 | barrier overhead 280 → 230 µs |

#1 单独可达 **~810 tok/s**，#1 + #2 + #3 可到 **~870 tok/s**。

---

## 9. 已知非瓶颈（已验证）

| 怀疑点 | 验证方式 | 结论 |
|---|---|---|
| ZMQ 进程间往返 | scheduler 不参与 decode 路径，只 prefill+cancel | 否 |
| per-step `cudaMalloc` | ForwardWorkspace 消除 | 否 |
| plan H2D 同步 | `upload_async` 已 async | 否 |
| D2H sync 阻塞 | `RUSTINFER_PROBE_NO_D2H=1` 测试，省 ~50 µs | 占 3.5% |
| Host launch overhead | cudaEvent 实测 25 µs | 占 1.7% |
| GEMM vs GEMV 选择 | `RUSTINFER_FORCE_GEMM=1` 反而更慢 | hgemv 是对的 |

---

## 10. 路径外的全链路 benchmark

如果要测 server 端聚合 throughput（多 batch、调度开销、tokenization 都
算上），用 `bench/bench_batch_throughput.py`：

```bash
python3 bench/bench_batch_throughput.py \
  --host 127.0.0.1 --port 8000 \
  --label llama3-1b \
  --max-tokens 128 \
  --batches 1,2,4,8,16,32 \
  --dataset bench/bench_prompts.json
```

输出格式：

```
| batch | agg tok/s | per-req mean tok/s | p50 latency |
|------:|----------:|-------------------:|------------:|
|    1  |     397.4 |              397.5 |       0.32s |
|    2  |     719.0 |              360.7 |       0.35s |
|    4  |   1,351.8 |              338.8 |       0.38s |
...
```

> 注意：单 batch agg tok/s 比 demo bin 低（397 vs 685），原因是 server
> 端有 HTTP/JSON 序列化、tokenizer detokenize 流式、scheduler tick 周期。
> 想测**纯 GPU 吞吐用 demo bin**；想测**真实生产路径用 server bench**。
