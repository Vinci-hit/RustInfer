# Worker Burst Decode Consistency Incident

## 摘要

2026-06-18 调查 worker 在极短时间内收到 `b=32` 并发 completion 请求时，同一 prompt 的 greedy 输出发生分叉的问题。表面现象是部分请求输出重复坏 token，例如“写写写”“泡泡泡”“排序排序”，并且同一批请求出现多个 completion hash。

最终确认有两个独立因素：

1. Worker 在多个 queued prefill command 之间插入 decode，使同一批 burst 请求按不同 decode batch shape 起步。
2. cuDNN frontend paged decode attention 在 serving replay 路径下存在 row-to-row 数值漂移，足以让 greedy decode 分叉。

本次修复：

- Worker 先 drain 当前 prefill cohort，再进入 decode。
- 在 prefill backlog 清空后增加 1ms bounded quiet window，收集同一短 burst 内随后到达的 prefill。
- 默认保留 cuDNN paged decode attention 作为性能路径；严格一致性/排障时可用 `RUSTINFER_DISABLE_CUDNN_ATTENTION=1` 切到 Flash fallback。

## 第一性原理

对本问题，最基础的不变量是：

> 对相同模型、相同权重、相同 prompt、相同 decode 参数、greedy sampling 的请求，如果没有随机采样，worker 必须产生相同 token 序列。

这个不变量需要两个层面同时成立：

1. **调度层不变量**

   同一短 burst 内的相同请求，不应因为 worker 内部抢跑而进入不同的 decode 时间线。否则第 0 个请求可能先以 batch=1 decode，一段时间后第 1 个请求以 batch=2 加入，后续请求再以 batch=N 加入。即使每个 kernel 都“基本正确”，不同 batch shape 的数值路径也可能让 greedy argmax 在边界处选择不同 token。一旦第一个 token 分叉，后续 KV cache 就进入不同状态，错误会被永久放大。

2. **kernel 后端不变量**

   同一个 decode step 的每一行请求，如果输入 token、position、KV 内容等价，attention 后端和 argmax 后端必须输出等价 logits/argmax。任何 row-to-row 漂移只要跨过 argmax 边界，就会在 greedy decode 中变成可见文本分叉。

因此，正确修复不能只看“请求成功率”，而要验证同一批相同请求的 token/hash 一致性。

## 起因

用户报告在 `b=32` 并发，即很短时间内发送 32 个请求时，出现类似结果：

```text
majority_hash=d610f982b3a5 count=6
[MISMATCH] #00 text=的代码法，，写写写写...
[MISMATCH] #01 text=的代码话的的的一段一段...
[MISMATCH] #02 text=的代码过程过程排序排序...
[MISMATCH] #03 text=的代码排序排序，泡泡泡泡...
[MISMATCH] #04 text=的代码泡泡泡泡冒冒冒...
```

本地复现：

```bash
.venv/bin/python bench/warmup_instruct.py \
  --url http://127.0.0.1:8000 \
  --warmup 0 \
  --batch-size 32 \
  --max-tokens 256
```

复现结果出现 `unique_outputs=11`，与用户报告的多 hash / 重复坏 token 模式一致。

## 调查经过

### 1. 追踪 worker 推理链路

请求进入 worker 后的主链路：

```text
serve_loop
  -> drain_data
  -> handle_prefill
  -> ActiveSeqMap / DecodeRows
  -> DecodeEngine::run_step
  -> ModelRunner::step_decode_abc_compact
  -> CUDA graph replay
  -> attention / argmax
  -> merge_compact_decode
```

关键状态：

- `ActiveSeqMap` 保存每个 sequence 的 `last_token`、`kv_len`、`block_table`。
- `DecodeRows` 保存 buffer A 的物理 row order，避免直接依赖 `HashMap` 迭代顺序。
- `step_decode_abc_compact` 让 CUDA graph 从 buffer A 读 token，写 buffer C，再由 merge kernel 把未完成 token compact 回 A。

`DecodeRows` 和 compact row mapping 没有发现明显错位，问题更像是请求进入 decode 的 batch/time shape 不一致，或 attention 后端在同 shape 内产生 row 分叉。

### 2. 隔离 cuDNN attention

使用环境变量禁用 cuDNN attention：

```bash
RUSTINFER_DISABLE_CUDNN_ATTENTION=1 ./target/release/rustinfer-worker --config rustinfer.toml
```

禁用 cuDNN 后，问题没有完全消失：

- `b=8, max_tokens=64` 仍出现多个输出。
- `b=32, max_tokens=64` 仍出现多个输出。

这说明 cuDNN 不是唯一根因，worker 调度也在制造分叉条件。

### 3. 检查 prefill/decode interleave

原 `serve_loop` 在多个 pending prefill command 之间插入 decode：

```text
prefill request 0
decode active batch=1
prefill request 1
decode active batch=2
...
```

这违反了同一短 burst 应作为一个 cohort 起步的调度不变量。删除 interleaved decode 后，`b=8` 从多个坏分支收敛到只剩第一个请求分叉，说明方向正确，但第一个 prefill 仍可能在后续 prefill 到达 worker 前先进入 decode。

### 4. 增加 prefill quiet window

继续加入 bounded quiet window：

- 只在刚处理过 prefill backlog 后生效。
- 如果 data socket 已空，最多等 1ms。
- 最多 16 轮 drain，防止持续流量无限阻塞 decode。

这个不是随意 sleep，而是 admission batching policy：在事件驱动系统里，如果希望短 burst 形成 cohort，就必须定义一个“何时认为当前 burst 已经到齐”的边界。这里的边界是 1ms quiet window。

### 5. 再次验证 cuDNN 默认路径

调度修复后，在禁用 cuDNN 的 Flash fallback 路径下：

- `b=8, max_tokens=64`：`unique_outputs=1`
- `b=32, max_tokens=64`：`unique_outputs=1`
- `b=32, max_tokens=256`：`unique_outputs=1`

但是重新启用默认 cuDNN attention 后：

- `b=32, max_tokens=256`：仍出现 `unique_outputs=10`

这证明第二个问题独立存在：cuDNN frontend paged decode attention 在当前 serving replay 路径下不满足 greedy 一致性不变量。

## 修复内容

### 1. Worker 调度修复

文件：

```text
crates/infer-worker/src/application/serve_loop.rs
```

改动：

- 删除 queued prefill 之间的 interleaved decode。
- 当前 drain 到的 prefill 先全部执行。
- 每轮处理完后继续 `drain_data`。
- 如果 data socket 为空，调用 1ms `wait_for_prefill_quiet`。
- 最多 16 轮，避免持续 prefill 流量饿死 decode。

这是结构性修复。它恢复了一个明确不变量：同一短 burst 的请求应尽量作为一个 active decode cohort 起步，而不是在 worker 内部被拆成 `1,2,3,...` 的 decode shape。

### 2. cuDNN paged decode 策略

文件：

```text
crates/infer-worker/src/infrastructure/cuda/kernels/attention_paged.rs
```

改动：

- 默认尝试 cuDNN paged decode attention。
- 保留 `RUSTINFER_DISABLE_CUDNN_ATTENTION`，用于严格一致性回归、排障和切换到 Flash fallback。
- `RUSTINFER_STRICT_CUDNN_ATTENTION` 仍可强制走 strict cuDNN 路径。

当前产品验收标准调整为：允许大模型在 greedy 下因后端数值路径不同产生合理漂移，但不允许复读、乱码或明显退化。因此 cuDNN 作为默认性能路径保留；Flash fallback 是严格一致性和排障基线。

## 验证结果

修复前：

```text
b=32, max_tokens=256
unique_outputs=11
出现 “写写写 / 泡泡泡 / 排序排序” 等坏分支
```

禁用 cuDNN 但未完整修复调度时：

```text
b=8, max_tokens=64
unique_outputs=6

b=32, max_tokens=64
unique_outputs=6
```

删除 interleaved decode 但未加 quiet window 时：

```text
b=8, max_tokens=64
unique_outputs=2
第 0 个请求仍可能抢先 decode 后分叉
```

调度修复 + Flash fallback：

```text
b=8, max_tokens=64
unique_outputs=1

b=32, max_tokens=64
unique_outputs=1

b=32, max_tokens=256
unique_outputs=1
```

调度修复 + cuDNN attention 开启：

```text
b=32, max_tokens=256
unique_outputs=10
```

默认性能路径修复后：

```text
b=32, max_tokens=256
32/32 succeeded
unique_outputs=10
未再出现 “写写写 / 泡泡泡 / 排序排序” 这类退化复读
```

严格一致性路径修复后，即 `RUSTINFER_DISABLE_CUDNN_ATTENTION=1`：

```text
b=32, max_tokens=256
32/32 succeeded
unique_outputs=1
completion_tokens=256
majority_hash=727817381453 count=32
```

构建验证：

```bash
cargo build --release -p infer-worker
rustfmt --check crates/infer-worker/src/application/serve_loop.rs \
  crates/infer-worker/src/infrastructure/cuda/kernels/attention_paged.rs
```

全仓 `cargo fmt --check` 当前仍会因为本次未触碰的 frontend/server/scheduler 既有格式差异失败，不作为本事故修复的通过条件。

## 这是不是彻底修复

从“不复读、不退化”的用户可见正确性目标看，默认路径已经修复：

- 短 burst 不再在 worker 内部被拆成不同 decode 起跑线。
- 默认 cuDNN attention 路径仍可能产生多 hash，但不再复现原始退化复读。
- 原始 `b=32, max_tokens=256` 场景不再出现“写写写 / 泡泡泡 / 排序排序”等坏分支。

从“同 batch 同 prompt 必须同 hash”的严格一致性角度看，默认 cuDNN 路径不是根治：

- 本次没有证明 cuDNN frontend paged decode 的内部漂移具体来自 plan cache、CUDA graph capture、paged table、workspace 复用，还是算法本身。
- 因此严格一致性测试应关闭 cuDNN，使用 Flash fallback 作为基线。

这不是“屎山补丁”的原因：

- 调度修复来自明确的不变量：相同 burst 请求必须作为 cohort 进入 decode。
- quiet window 是 admission policy 的边界，不是掩盖错误的随机 sleep。
- cuDNN 策略被明确拆成性能容忍模式和严格一致性模式，而不是混淆两个验收标准。
- Flash fallback 保留为可验证的正确性基线。

## 后续建议

1. 增加自动化一致性回归

   覆盖：

   - `b=1, 2, 3, 4, 8, 16, 24, 32`
   - `max_tokens=64/256`
   - CUDA graph replay 开关
   - cuDNN attention on/off
   - strict 模式下 completion text hash 必须全 batch 一致
   - performance 模式下允许多 hash，但必须检测复读、乱码、明显退化

2. 给 cuDNN attention 单独建立 degeneration benchmark

   cuDNN 默认保留性能路径，但需要单独验证：

   - 多轮 b=32/b=64 burst 压测无复读退化。
   - 文本不出现明显乱码或低质量循环。
   - graph replay 与 eager 路径的质量差异可接受。
   - 若未来要求严格一致，再补 row logits/argmax 一致性测试。

3. 把 admission quiet window 参数化

   当前 1ms 是保守默认。后续可以放入配置，例如：

   ```toml
   worker_prefill_quiet_ms = 1
   worker_prefill_drain_rounds = 16
   ```

   这样低延迟优先和吞吐/一致性优先的部署可以显式选择策略。

4. 记录 worker capability

   Worker Ready 可以上报 attention backend，例如 `flash_paged_decode` 或 `cudnn_paged_decode`，方便 scheduler 和监控系统理解当前正确性/性能档位。
