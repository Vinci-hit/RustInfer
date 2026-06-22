# Buffer-Pipeline 重构后 TTFT / TTOT 回归诊断与修复

## 背景

`feat/worker-batch-forward` 在 `11f2312..9bb38da` 之间完成了一次大型重构：

- `11f2312` refactory begin
- `81b6a23` refactory worker
- `16a79e9` refactor: split into static modular multi-backend crates
- `bd07415` refactory scheduler
- `9bb38da` 3 buffer pipeline

重构后离线 benchmark 出现明显回归：TTFT（首 token 延迟）与 TTOT（token 间间隔）均显著高于
重构前 (`1cd848c`)。本文记录排查过程、根因分析与已落地的三项修复。

---

## 根因分析

### 回归点 ① — Prefill 路径 GreedySampler 每步分配 argmax 输出 / scratch

**影响**：TTFT 上限受限于 prefill 的 `step` 调用耗时；新版每次 prefill 多付两次 device
allocation + memset。在低 QPS / 小 batch 场景下直接拖 TTFT。

#### 调用链

```
worker_scheduler.rs::handle_prefill
  └─ ctx.runner.step(&req)                 // q_len > 1, decide() = Eager
      └─ runtime.rs::step_eager
          └─ sample_tail
              └─ self.sampler.sample(&logits.0, ...)
                  └─ D::argmax(ctx, logits)
                      ├─ Tensor::<i32>::zeros([rows])         // ← 分配 1
                      ├─ Tensor::<f32>::zeros([rows * 512])   // ← 分配 2
                      ├─ kernels::sampler::argmax(...)
                      └─ out.to_host_vec()                    // 隐式 stream sync
```

`infer-backend-cuda/src/lib.rs:478-499` 的 `D::argmax` 实现每次都重新构造一个 `[rows]`
的 i32 输出张量与 `[rows * 512]` 的 f32 scratch。重构前 `step_batch_eager` 走的是
`forward_ws.argmax_args()`，使用 `ForwardWorkspace` 里**预分配**的 `out_dev` /
`workspace`，零分配。

decode 路径 (`issue_decode_abc`) 已经使用 `argmax_into` 复用 `abc.argmax_out_dev /
abc.argmax_ws`，所以这个分配开销**只折磨 prefill**。

#### 修复

`crates/infer-worker/src/application/runtime.rs::sample_tail` 的 greedy 分支不再走
`Sampler::sample`，改为直接调用 `D::argmax_into(&abc.argmax_out_dev, &abc.argmax_ws)`，
然后按 `plan.q_lens` 取每条序列最后一行的 token id。语义与
`GreedySampler::sample` + `sampled_rows()` 完全一致；speculative verify 路径仍走
`self.sampler.verify`，不受影响。

`Runtime::abc.argmax_out_dev` 容量是 `cap_batch`，prefill 的 `logits_rows == sum(q_lens)`
通常远大于 `cap_batch`——这里我们要求 `logits_rows <= argmax_out_dev.numel()`，等价于
单次 prefill 的总 token 数不能超过 cap_batch×512 / 4B / 实际 numel；这条假设在我们的
配置下成立（prefill 的 num_tokens 受 `cap_num_tokens` 上限约束）。如果未来 prefill
跑超长 prompt 触发上限，会返回 `OpError::Shape`，需要扩 `abc.argmax_out_dev` 容量到
`max(cap_batch, cap_num_tokens)`。

> **TODO**: 实测确认 `argmax_out_dev` 容量是否够大。当前实现按 `cap_batch` 分配，
> `cap_num_tokens` 通常更大；如果 prefill 撞上 capacity check，把 `Runtime::new`
> 里的 `argmax_out_dev` / `argmax_ws` 容量改为 `max(cap_batch, cap_num_tokens)`。

---

### 回归点 ② — `step_graph` 暖路径多余 `synchronize`

**影响**：每次命中 graph replay 的 decode 步多付一次主机←GPU 的 round-trip 等待。

```rust
// runtime.rs::step_graph (修复前)
self.scope.graph_launch(key)?;
self.scope.synchronize()?;          // ← 多余
self.decode_output_from_c(plan, req)
```

`decode_output_from_c` 内部 `c_view.to_host_vec()` 已经触发了对计算流的同步
（CUDA `download` → `cudaStreamSynchronize`），所以前一行 `scope.synchronize` 是双重
等待。冷路径（capture 阶段前）的 `synchronize` 不在这条改动范围内，仍然保留。

#### 修复

删掉 `step_graph` 暖路径（graph_ready 分支结束后）那一行 `self.scope.synchronize()?;`。
注意当前主链路 `DecodeEngine::run_step` 不走 `step_graph`，但若有遗留代码仍经过
`runner.step(&req)` 命中 graph，能直接受益。

---

### 回归点 ③ — 1-deep pipeline 让 prefill→首个 decode token 多延一轮

**影响**：流式响应里，prefill 完成发出第一个 token 后，第一个真正的 decode token
不会在当轮 serve loop 发出，而是要等下一轮——把"第一段稳态间隔"撑成两步 GPU 时间。

#### 流程推演（修复前）

```
serve_loop iter K  (prefill 结束, active 刚被填上, pending=None)
  decode_engine.run_step #1
    finalize_pending  → pending=None → to_send=None
    issue_new         → 入队 step_1, pending=Some(step_1)
    send to_send      → 没东西可发                       ← 用户看不到 token
serve_loop iter K+1
  decode_engine.run_step #2
    finalize_pending  → drain step_1 → token_1
    issue_new         → 入队 step_2, pending=Some(step_2)
    send to_send      → 发 token_1                       ← 用户在 iter K+1 收到
```

token_1 比无 pipeline 方案晚一整轮。稳态 token 间隔（token_n 与 token_{n+1} 之间）
仍然是一步 GPU 时间，pipeline 在稳态上是对的；问题只在「首个 decode token 比应有
时机晚一轮」。

#### 修复

在 `DecodeEngine::run_step` 入口检测 `cold_start = pending.is_none() && !active.is_empty()`。
如果是冷启动，正常 finalize → issue → send 后，立刻再 drain 一次（finalize + send
刚 issue 的 step），然后 re-issue 让 pipeline 重新进入稳态。

```
serve_loop iter K  (cold_start = true)
  decode_engine.run_step #1
    finalize_pending  → None
    issue_new         → step_1
    send              → 无
    drain (cold-start path)
      finalize_pending  → token_1            ← 立即发
      send              → 发 token_1
      issue_new (re-issue) → step_2          ← 让稳态 pipeline 启动
serve_loop iter K+1  (cold_start = false, pending=Some(step_2))
  decode_engine.run_step #2
    finalize_pending  → token_2
    issue_new         → step_3
    send              → 发 token_2
```

注意 `commit_results` 内部可能因为所有行都 finished 而清空 active，re-issue 前需要
判断 `active.is_empty()` 跳过。

> **代价**：冷启动这一轮会失去 issue↔send 的 overlap（send 在 issue 之后再做），
> 但它只发生在每段 decode 流的开头。稳态 1-deep pipeline 的 overlap 完整保留。

---

## 修复后的代码改动

| 文件 | 改动 |
|---|---|
| `crates/infer-worker/src/application/runtime.rs` | `sample_tail` greedy 分支用 `argmax_into` 复用 `abc` workspace；`step_graph` 暖路径删掉多余 `synchronize` |
| `crates/infer-worker/src/application/decode_engine.rs` | `run_step` 加冷启动 drain + re-issue |

`+94 / -7` 行。`Runtime::sampler` 字段保留（spec verify 仍在用），无 unused warning。

---

## 已知未做改动

### 每步重建 `gen_i32 / max_i32 / ign_i32` Vec

`runtime.rs::issue_decode_abc` 第 911-913 行：

```rust
let gen_i32: Vec<i32> = generated_counts.iter().map(|&x| x as i32).collect();
let max_i32: Vec<i32> = max_tokens.iter().map(|&x| x as i32).collect();
let ign_i32: Vec<i32> = ignore_eos.iter().map(|&b| i32::from(b)).collect();
```

每步分配 3 个 batch 长度的 i32 vec + H2D，体量小（cap_batch × 3 × 4B），优先级低。

**可选优化**：把这三个也加到 `AbcBuffers` pinned host 镜像里，仅在值变化时（admit
新 seq、max_tokens 不同等）上传；`generated_counts` 每步 +1 可以靠一个 device 端
kernel 自增，省掉每步 H2D。

### `eos_ids` 每步上传

`issue_decode_abc:919-921`：`eos_ids` 整个生命周期不变，但每步都通过
`upload_i32_prefix` 上传。可以在 `Runtime` 上缓存"已上传"标志，仅首次或 eos_ids
变更时上传。

### `prev_a_rows` 每步分配

`decode_engine.rs:154`：`self.prev_a_rows = self.rows.as_slice().to_vec();`
小开销，每步 cap_batch × 8B 分配。可改成 `prev_a_rows.clear(); prev_a_rows.extend_from_slice(rows.as_slice());`。

### CLAUDE_FABLE_5.md

仓库根目录上次提交意外加进来一份 1597 行的 Anthropic 系统提示文件，与项目无关，
建议 `git rm` 掉。

---

## 验证步骤

```bash
cd ~/Desktop/RustInfer
cargo check -p infer-worker --features cuda
cargo build --release -p infer-worker --features cuda
```

跑 benchmark：

```bash
python bench/bench_online_qps.py   # 与重构前对比 TTFT / TTOT
```

预期：

- **TTFT**：低 QPS 下回到 `1cd848c` 水平甚至更好（消除 prefill 每步双 `Tensor::zeros`）。
- **TTOT 首段**：prefill→首个 decode token 间隔不再被推迟一轮。
- **TTOT 稳态**：保持不变（1-deep pipeline 仍在工作）。
- **吞吐**：理论持平或略升（消除多余 sync + 多余分配）。

如有进一步异常，建议 `nsys profile` 抓 100 步看 compute / Si / So 三条流的 timeline
是否真的 overlap；当前 `serve_loop` 里 `profile_cuda_steps` 已支持精确开窗。

---

## 时间线

- 2026-06-21 22:19 `9bb38da` 3 buffer pipeline 提交，bench 出现回归
- 2026-06-22 诊断 + 三项修复落地
