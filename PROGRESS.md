# RustInfer 端到端跑通进展

> 最后更新：2026-05-27
> Branch: `feat/worker-batch-forward`

## 一、端到端跑通 ✅

完整链路已在 H20（CUDA 13.1）上验证：

```
HTTP → infer-server → ZMQ → infer-scheduler (paged KV) → ZMQ → infer-worker (CUDA) → ZMQ ↩
```

- ✅ Llama-3.2-1B-Instruct (bf16)
- ✅ Qwen3-4B-Instruct (bf16, sharded safetensors)
- ✅ paged KV pool（block_size=16；与 scheduler 协议一致）
- ✅ 单 seq / 多 seq ragged batch
- ✅ chat template + system prompt
- ✅ **CUDA Graph 加速 decode-only 步骤**（capture sizes ≤ cap_batch；padding+replay）

事实正确性验证：
- "The capital of France is" → " Paris. The Eiffel Tower is..." (Llama)
- "What is the capital of France? Answer in one word." → "Paris" (Qwen3)
- 输出 token id 序列与 transformers reference byte-exact 一致
- CUDA Graph replay 与 eager 输出 token 序列一致

## 一·五、CUDA Graph 集成（Phase 5f）✅

**架构**:
- `domain/forward_workspace.rs`：12 个长寿命 forward 中间张量 + flash-decode scratch + argmax_out_dev/host
- `domain/batch_workspace.rs`：长寿命 plan 张量 + host staging Vec；`build_plan` 异步上传 + 返回 `view_raw` 共享
- `MemoryPort::upload_async`：`cudaMemcpyAsync` 不带 sync
- `argmax_batched_decode_into`：dense `[batch, vocab]` 单 launch + 直写设备 i32 输出（零 D2H、零 alloc）
- `app/cuda_graph_runner.rs::warmup_and_capture_all`：倒序（16→8→4→2→1）、每 size 2 轮 warmup + capture + launch validate
- `infra/cuda/config.rs::CudaConfig.graphs`：`Mutex<HashMap>` 包装，使 `&CudaConfig` 即可 capture/launch
- `app/model_runner.rs::prime_graphs_cuda`（Cuda-only impl）：保留最后 1 个物理块作为 graph scratch；自动剔除超 cap_batch 的 size
- `step_batch_with_graph`：decode-only + batch ≤ max_capture_size 时走 graph，否则 eager fallback；padding 用 scratch block

**关键不变量**：所有 forward intermediate 在 ModelRunner 构造时一次性 alloc，`view_raw` 仅做 Arc 共享 → graph 捕捉到的是稳定地址。

**A/B 验证**（Llama-3.2-1B，batch=1，1000 tokens decode）：

| 模式  | tok/s |
|------|------:|
| eager | 595.3 |
| graph | 622.4 |

| 模式  | tok/s（200 tok 平均）|
|------|------:|
| eager | ~570 |
| graph | ~590 |

env vars：
- `RUSTINFER_DISABLE_GRAPH=1` 关闭 graph，走 eager 对照
- `RUSTINFER_TRACE_GRAPH=1` 打印每次 replay 的 slot

## 二、Throughput 基准（H20，bf16，max_tokens=128，batch=concurrency）

每个 batch 大小下，N 个并发请求在 server 端同时入队，agg = 总生成 tokens / wall-clock。

### Llama-3.2-1B-Instruct

| batch | agg tok/s | per-req mean tok/s | p50 latency |
|------:|----------:|-------------------:|------------:|
|    1  |     397.4 |              397.5 |       0.32s |
|    2  |     719.0 |              360.7 |       0.35s |
|    4  |   1,351.8 |              338.8 |       0.38s |
|    8  |   2,295.3 |              287.4 |       0.45s |
|   16  |   2,282.0 |              145.5 |       0.84s |
|   32  |   3,772.9 |              118.5 |       1.05s |

### Qwen3-4B-Instruct

| batch | agg tok/s | per-req mean tok/s | p50 latency |
|------:|----------:|-------------------:|------------:|
|    1  |     149.5 |              149.5 |       0.86s |
|    2  |     269.0 |              135.0 |       0.95s |
|    4  |     307.1 |               76.9 |       1.67s |
|    8  |     926.0 |              115.9 |       1.11s |
|   16  |     832.9 |               52.1 |       2.46s |
|   32  |   1,232.3 |               38.5 |       3.32s |

数据来源：`bench/bench_batch_throughput.py`（dataset = `bench/bench_prompts.json`，51,906 条 Alpaca-style prompts）。

> 备注：batch=4 / 16 / 32 在 Qwen3-4B 上有抖动，可能与 paged attention kernel 在特定 batch size 退化或 chunked-prefill 调度有关。下一步分析。

## 三、关键修复（按时间顺序）

| 修复 | 影响 |
|---|---|
| `swiglu` kernel 自带 silu，去掉模型层重复 silu | 修复 Llama 输出乱码 |
| EOS early stop（128001/128008/128009） | "Paris.<EOS>" 正确终止 |
| `attn_out` shape 用 `q_dim` 而非 `dim` | 修复 Qwen3-4B（q_dim=4096 ≠ dim=2560）输出乱码 |
| `SafetensorsReader` 支持 sharded layout | Qwen3-4B 多 shard 权重加载 |
| `serve_loop`: scheduler `SchedulerHello` 跳过 + heartbeat 1s | 修复 worker liveness timeout |
| `OpBackend::scatter_kv_paged` / `attention_paged` | 全面切换到 paged KV，配合 scheduler 协议 |

## 四、命令参考

```bash
# Llama-3.2-1B
cd /root/RustInfer
PATH=$PWD/target/release:$PATH ./target/release/rustinfer-server \
  --model /apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b \
  --model-type llama3 --device cuda:0 \
  --max-batch-tokens 8192 --max-batch-seqs 64 --max-model-len 4096 --port 8000

# Qwen3-4B
PATH=$PWD/target/release:$PATH ./target/release/rustinfer-server \
  --model /apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct \
  --model-type qwen3 --device cuda:0 \
  --max-batch-tokens 8192 --max-batch-seqs 64 --max-model-len 4096 --port 8000

# Throughput sweep
python3 bench/bench_batch_throughput.py --label <name> --max-tokens 128 --batches 1,2,4,8,16,32
```
