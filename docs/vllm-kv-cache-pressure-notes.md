# vLLM KV Cache 压力处理经验

本文记录从 vLLM V1 观察到的 KV cache 压力处理经验，并转换成 RustInfer 后续实现和压测时可以直接对照的设计准则。

## 核心结论

vLLM 在 KV cache 快满时并不是“无感扩容”，也不是依赖 GPU OOM 后恢复。它的稳定性主要来自 scheduler 层的容量控制：

- KV cache 在启动时预分配为固定大小的 block pool。
- 每个调度 step 只给请求分配当前能放下的 token / block。
- 新请求进不来就留在 waiting queue，而不是继续挤进 GPU。
- running 请求续写时如果分配失败，scheduler 会 preempt 低优先级或队尾请求，释放它的 KV block。
- vLLM V1 默认用 recompute 处理 preemption，不再依赖 GPU/CPU KV swap。
- prefix cache 里没有 live owner 的 block 是可驱逐候选，可以被新请求复用。

所以 60s 压测里“第 30s 理论上 KV cache 要满，但吞吐/延迟还稳定”，通常不是因为 KV cache 没有边界，而是系统已经进入了一个稳定工作点：活跃请求占住固定 KV 容量，新请求排队，完成请求释放 block，必要时少量请求被抢占重算。

## 不要按累计 token 估算 KV 是否会满

KV cache 的压力来自“当前仍然活跃且需要被 attention 访问的 token”，不是压测开始以来累计处理过的 token。

错误估算：

```text
60s 压测总生成 token 数 / KV cache 容量
```

更合理的估算：

```text
sum(active_request_prompt_tokens + active_request_generated_tokens)
```

再乘上每 token KV 大小：

```text
layers * kv_heads * head_dim * 2(K,V) * dtype_bytes
```

分页实现里还要考虑 block_size 造成的向上取整和内部碎片。

## vLLM 的实际调度路径

vLLM V1 的 scheduler 没有严格区分 prefill phase 和 decode phase。它按 request 的 `num_computed_tokens` 和目标 token 数计算每个 step 还需要补多少 token。

简化流程：

1. 先调度 RUNNING 请求。
2. 对每个 RUNNING 请求计算本 step 的 `num_new_tokens`。
3. 调用 `kv_cache_manager.allocate_slots()` 分配新增 KV block。
4. 如果能分配，本 step 正常执行。
5. 如果不能分配，scheduler 选择 victim 做 preemption。
6. 被 preempt 的请求释放 KV block，状态回到 waiting/preempted，之后有空间再重算。
7. 如果本 step 发生了 preemption，通常不再继续吸纳新的 waiting 请求。

这个路径有两个关键性质：

- GPU worker 不应该自己决定牺牲谁；preemption 决策在 scheduler。
- 分配失败必须在调度层变成“少调度、排队、驱逐、抢占”，不能变成 GPU OOM。

## Prefix Cache 与 LRU 驱逐

vLLM 的 block pool 里，free queue 同时包含真正空闲的 block 和 prefix cache 中 ref_cnt 为 0 的可驱逐 block。新分配需要复用这些 block 时，会先清理旧的 block hash 元数据，再把 block 分给新请求。

这带来一个重要压测现象：

- `kv_cache_usage` 看起来接近 100% 不一定说明系统马上失败。
- 如果大量 block 是未被 live request pin 住的 prefix cache，它们仍然可以被驱逐复用。
- 真正危险的是 live-owned block 占满，且 waiting/running 的新增 token 还需要继续增长。

RustInfer 的 RadixTree LRU 不应该释放 live owner 持有的 slot。当前代码里的核心不变量是正确方向：只有 owner 为空的 node 才能进入 LRU。

## Preemption 不是免费能力

vLLM 文档也明确说明：preemption/recompute 会影响端到端延迟。平均吞吐稳定不代表没有代价，代价常体现在：

- 个别请求 E2E latency 变长。
- p99 / p999 抬升。
- TTFT 在高压下变差。
- 被 preempt 请求重复 prefill 或重复计算一段 KV。

因此压测报告不要只看总吞吐和 p50。必须同时看 preemption 次数和长尾延迟。

## RustInfer 设计准则

### 1. Admission control 要发生在 scheduler

RustInfer 不应该把所有 HTTP 请求直接推进 worker，让 worker 在 `alloc_indices()` 失败后才处理容量问题。worker 的 `AllocFailed` 应该是兜底信号，而不是常态调度路径。

Scheduler 侧需要维护或估算：

- 总 KV slots / blocks。
- live-owned slots。
- LRU 可回收 slots。
- 本 step 新增 token 需要的 slots。
- waiting / running 请求数量。

当预计放不下时，优先选择：

1. 不吸纳新的 waiting 请求。
2. chunk prefill，减小单 step KV 增量。
3. 驱逐 RadixTree LRU。
4. 仍不足时 preempt victim。

### 2. Running decode 优先于新 prefill

vLLM 的稳定 ITL 很大程度来自优先推进已有 running 请求。RustInfer 在高压下也应优先保证 decode token 的连续产出，避免长 prefill 抢占 decode 的 token budget 和 KV budget。

实践规则：

- 先给 active decoding 请求分配本 step decode token。
- 再用剩余 token budget 做 chunked prefill。
- 长 prompt 必须可切块，不能一次性独占 `max_batch_tokens`。
- 高 KV 压力下暂停吸纳新 prefill，比让所有请求一起变慢更稳定。

### 3. Worker 只报告分配失败，不做策略决策

当前协议方向是合理的：

- worker `alloc_indices(n)` 失败后发送 `AllocFailed { shortfall, round }`。
- round 0：scheduler 做 RadixTree LRU eviction，回复 `FreeKvIndices`。
- round 1：scheduler 做 victim preemption，回复 `Preempt(sequence_ids, free_indices)`。

需要保持这个边界：worker 是被动执行释放/丢弃，scheduler 负责决定释放哪些 slot、抢占哪些 request。

### 4. Preempt 后必须能正确重算

被 preempt 的请求不能被当成正常失败。它应该回到可调度队列，后续有空间时重新 prefill/重算必要 KV。

需要保证：

- RequestTable 中状态转换明确：Running/Decoding -> Preempted/Queued。
- preemption_count 增加，便于指标观测。
- 清理 worker active state 和 block_table。
- 清理或修正 RadixTree owner，避免 stale owner 阻止 LRU。
- 如果有 prefix cache 命中，恢复时可以复用仍然有效的 prefix。
- 如果没有可复用 KV，则从 prompt 重新计算，不返回错误给用户。

### 5. 不要依赖 heartbeat 做 KV 压力控制

KV 压力应该由真实分配路径触发，或由 scheduler 的本地预算预测提前处理。heartbeat 适合做 liveness，不适合做容量控制。

原因：

- heartbeat 粒度太粗，容易晚于实际分配失败。
- 高频 heartbeat 会引入无效控制流。
- 分配失败和调度计划才知道真正 shortfall。

## 压测时如何判断系统是否健康

### 健康的高压稳态

这些现象通常可以接受：

- KV usage 接近 1.0，但没有 OOM。
- running 数稳定在上限附近。
- waiting 数随到达率上升而增加。
- preemption 很少或为 0。
- ITL 稳定，TTFT 随 queue 增长而变大。
- 请求结束后 KV slots 能持续回收。

### 需要警惕的现象

这些现象说明容量策略可能有问题：

- `AllocFailed` 高频出现，说明 admission control 太晚。
- round 0 LRU eviction 经常为空，但系统仍持续吸纳新请求。
- preemption 持续增长，且 p99/p999 明显变差。
- waiting 不增长但请求 timeout，可能有请求丢失或状态卡死。
- KV usage 接近 1.0 后吞吐突然归零，可能是 worker 等待释放路径死锁。
- 被 preempt 请求没有重新进入队列，最终静默丢失。
- prefix cache owner 没清干净，导致 LRU 看起来为空但实际有可回收链。

## 建议指标

RustInfer 至少需要这些 Prometheus 或日志指标：

- `rustinfer_kv_usage_ratio`
- `rustinfer_kv_slots_total`
- `rustinfer_kv_slots_free`
- `rustinfer_kv_slots_live`
- `rustinfer_kv_slots_lru`
- `rustinfer_alloc_failed_total{round="0|1"}`
- `rustinfer_lru_evicted_slots_total`
- `rustinfer_preemptions_total`
- `rustinfer_preempted_requests_in_queue`
- `rustinfer_requests_running`
- `rustinfer_requests_waiting`
- `rustinfer_request_queue_time_seconds`
- `rustinfer_time_to_first_token_seconds`
- `rustinfer_inter_token_latency_seconds`
- `rustinfer_e2e_request_latency_seconds`

压测对比时，至少同时汇报：

- throughput tokens/s
- p50/p90/p99 TTFT
- p50/p90/p99 ITL
- p50/p90/p99 E2E latency
- preemptions total
- alloc failed total by round
- LRU evicted slots
- peak waiting queue length

## 调参方向

如果频繁 preempt：

- 增大 `mem_fraction_static` 或显式增大 `num_blocks`。
- 减小 `max_batch_seqs`。
- 减小 `max_batch_tokens`，降低单 step 峰值 KV 增量。
- 启用/加强 chunked prefill。
- 降低 `max_model_len`，避免为过长请求预留过大潜在容量。
- 检查 prompt 分布，长 prompt 高并发更容易制造 live KV 压力。
- 对 prefix cache 命中率低的压力测试，不要期待 LRU 能释放很多空间。

如果吞吐稳定但 TTFT 变差：

- 多半是 waiting queue 在吸收流量。
- 这是正常背压，但要确认没有 timeout 或 starvation。
- 可以增加并行资源、减少到达率、或提高 KV 容量。

如果 ITL 变差：

- decode 没有被优先保障。
- prefill chunk 太大。
- 单 step batch token 预算被长 prompt 占满。
- preemption/recompute 太频繁。

## RustInfer 压测建议

仓库已有 KV 压力脚本：

```bash
cd /root/RustInfer
uv run python bench/bench_kv_pressure.py \
  --url http://127.0.0.1:8000 \
  --concurrency 64 \
  --waves 2 \
  --prompt-lines 80 \
  --max-tokens 128 \
  --dump-json /tmp/rustinfer_kv_pressure.json
```

为了更容易触发压力，可以临时降低 `rustinfer.toml` 里的 KV 容量，例如减小 `mem_fraction_static` 或设置较小的 `num_blocks`。压力测试 prompt 已经把唯一标识放在开头，目的是避免 prefix cache 掩盖 KV 压力。

验证目标：

- 压测期间没有 panic / timeout / deadlock。
- `AllocFailed` round 0 能释放 LRU 时，worker 收到 `FreeKvIndices` 后继续执行。
- round 0 释放不足时，round 1 能 preempt victim。
- 被 preempt 请求不会丢失，后续能重新调度或明确返回可解释错误。
- 压测结束后健康检查仍然成功。

## 设计上的一句话原则

KV cache 快满时的“无感”不是靠隐藏错误，而是靠明确的背压和可恢复退让：

```text
少接新请求，先保 running；能驱逐就驱逐，不能驱逐就抢占；抢占后必须能重算恢复。
```
