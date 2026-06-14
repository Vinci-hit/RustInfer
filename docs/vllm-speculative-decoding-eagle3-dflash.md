# vLLM 投机解码设计调研：EAGLE3 与 DFlash

本文基于 `/root/vllm` 当前源码整理，只关注 vLLM V1 中 EAGLE3 和 DFlash 相关设计。目标是把 vLLM 的实现思路转成 RustInfer 后续实现 speculative decoding 时可直接对照的设计说明。

## 核心结论

vLLM 的 speculative decoding 不是在 worker 内部“偷偷多生成几个 token”，而是贯穿 scheduler、model runner、sampler、drafter 的跨 step 流水线：

1. 上一轮 drafter 生成 draft token。
2. scheduler 把这些 draft token 追加到请求的待计算 token 中，形成下一轮 target model 的输入。
3. target model 一次 forward 验证多个 draft token，并额外产生一个 bonus 位置的 logits。
4. rejection sampler 根据 target logits 决定接受多少 draft token，并在必要时恢复采样。
5. 同一轮 target 输出处理完成后，runner 立即调用 drafter 为下一轮生成新的 draft token。
6. scheduler 在下一次 step 使用这些新的 draft token。

设计目标是把大模型逐 token decode 中的 memory-bound 串行路径，转换成“大模型批量验证 + 小 draft 模型快速预测”的流水，同时通过 rejection sampling 保持目标模型分布不变。

## 关键源码入口

- 配置与自动识别：`vllm/config/speculative.py`
- 调度层注入 draft token：`vllm/v1/core/sched/scheduler.py`
- target forward、采样、调用 drafter：`vllm/v1/worker/gpu_model_runner.py`
- EAGLE/EAGLE3/DFlash proposer 基类：`vllm/v1/spec_decode/eagle.py`
- DFlash proposer：`vllm/v1/spec_decode/dflash.py`
- spec decode 元数据：`vllm/v1/spec_decode/metadata.py`
- rejection sampler：`vllm/v1/sample/rejection_sampler.py`
- EAGLE3 Llama draft 模型：`vllm/model_executor/models/llama_eagle3.py`
- EAGLE3 DeepSeek draft 模型：`vllm/model_executor/models/deepseek_eagle3.py`
- DFlash Qwen3 draft 模型：`vllm/model_executor/models/qwen3_dflash.py`
- DFlash 行为测试：`tests/v1/spec_decode/test_eagle.py`

仓库里还有一套更模块化的 `vllm/v1/worker/gpu/model_runner.py` 和 `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` 路径，它把 speculator/sampler/input batch 拆得更清楚。它对 EAGLE3 有显式支持，但 DFlash 的专用 cross-attention/precompute 路径主要仍在 `vllm/v1/spec_decode/dflash.py` 和旧的 `gpu_model_runner.py` 链路中体现。

## 配置层设计

`SpeculativeConfig` 把 `eagle3` 和 `dflash` 都归入 EAGLE family：

- `EagleModelTypes = ("eagle", "eagle3", "extract_hidden_states", MTP..., "dflash")`
- `use_eagle()` 对 `dflash` 也返回 true。
- `use_dflash()` 单独识别 `method == "dflash"`。

当用户传入 `speculative_config` 后，vLLM 会：

1. 创建 draft model 的 `ModelConfig`，runner 设置为 `draft`。
2. 如果用户没有显式写 `method`，根据模型名和 HF config 自动识别：
   - 模型名包含 `eagle3` -> `method="eagle3"`
   - 模型名包含 `dflash` -> `method="dflash"`
3. 对 `eagle/eagle3/dflash` 包一层 `EAGLEConfig`，把架构名重写成 `Eagle...`、`Eagle3...` 或 `DFlash...`，从而路由到对应 vLLM 模型类。
4. 对 `dflash` 强制 `parallel_drafting = True`。
5. 根据 `num_speculative_tokens` 生成默认 token tree：`[(0,), (0, 0), ...]`，即链式 draft。
6. 为 EAGLE3/DFlash 记录辅助 hidden state 配置，并把它加入编译 hash，因为目标模型 forward 的返回值会改变。

EAGLE3 和 DFlash 都需要 target 模型支持返回中间层 hidden states。配置校验里只允许一组支持 aux hidden state 的模型族，例如 llama、qwen、deepseek、kimi、gemma4 等。

## Scheduler 层：把 speculative token 当作待计算 token

vLLM V1 scheduler 没有严格区分 prefill/decode。每个请求只有：

- `num_computed_tokens`
- prompt token
- output token
- speculative token
- async output placeholder

调度时的核心公式是：

```text
num_new_tokens =
  request.num_tokens_with_spec
  + request.num_output_placeholders
  - request.num_computed_tokens
```

如果请求上已有 `request.spec_token_ids`，scheduler 会把本 step 能调度到的 speculative token 放入 `SchedulerOutput.scheduled_spec_decode_tokens`，并清空 request 上的旧 speculative token。target runner 后续会把这些 token 合并进 input ids。

采样完成后，scheduler 根据实际生成 token 数修正状态：

```text
num_draft_tokens = len(scheduled_spec_token_ids)
num_accepted = len(generated_token_ids) - 1
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
```

这里 `-1` 是因为生成结果包含 bonus/recovered 位置。被拒绝的 draft token 虽然已经进入本轮 target forward，但不会成为请求的真实输出，scheduler 必须从已计算 token 计数中扣掉它们。

## Target 验证与 rejection sampler

当 `scheduled_spec_decode_tokens` 非空时，`GPUModelRunner._prepare_inputs()` 会构造 `SpecDecodeMetadata`：

- `draft_token_ids`：flatten 后的 draft token。
- `num_draft_tokens`：每个请求有多少 draft token。
- `cu_num_draft_tokens`：draft token 累积和。
- `target_logits_indices`：target logits 中用于验证 draft token 的位置。
- `bonus_logits_indices`：所有 draft 都接受时可直接追加的 bonus 位置。
- `logits_indices`：本轮需要从 hidden states 取 logits 的位置，长度是 `sum(num_draft_tokens + 1)`。

target model 只对 `logits_indices` 位置计算 logits。随后 `RejectionSampler` 做两件事：

1. 对 bonus 位置按正常 sampler 采样。
2. 对 draft 位置执行 rejection sampling。

在 greedy 情况下，接受条件退化为 target argmax 和 draft token 是否相同。随机采样时，算法按 speculative sampling 的概率规则接受 draft token，拒绝后从修正分布恢复采样。最终输出形状是：

```text
[batch_size, max_spec_len + 1]
```

被拒绝位置用 `-1` 占位，`parse_output()` 再过滤成每个请求真正输出的 token 列表。

这个设计把“接受多少 token”的判断集中在 sampler，而不是 drafter。drafter 只负责提案，target logits 才是最终权威。

## EAGLE3 的设计理念

EAGLE3 的核心思路是：draft 模型不是另一个完整小 LM，而是一个利用 target hidden states 的轻量预测头。

它相比普通 draft model 的关键差异：

- target model forward 需要返回若干辅助层 hidden states。
- drafter 先把多个辅助 hidden states 拼接，再通过 `fc` 投影回 draft hidden size。
- drafter 第一层把 token embedding 和 target hidden state 拼接后做 attention/MLP。
- draft 模型通常只包含少量层，可共享 target embedding 和 lm_head 来节省显存。
- draft vocab 可能是 target vocab 的子集，通过 `draft_id_to_target_id` 映射回 target vocab。

EAGLE3 draft 模型中的典型结构：

```text
target aux hidden states
  -> concat
  -> fc/combine_hidden_states
  -> EAGLE3 decoder layers
  -> lm_head
  -> draft token ids
```

在 `llama_eagle3.py` 中，第一层的 qkv 输入维度是 `2 * hidden_size`，因为它会拼接 embedding 和 hidden state。DeepSeek 版本也遵循相同思路，只是 attention 使用 DeepSeek MLA，且 EAGLE3 层固定用 MLP 而不是 MoE。

## EAGLE3 执行流程

### 1. 初始化

`GPUModelRunner` 初始化 speculative decoding 时：

1. `method == "eagle3"` 时创建 `EagleProposer`。
2. 设置 `use_aux_hidden_state_outputs`，让 target model forward 返回 `(hidden_states, aux_hidden_states)`。
3. drafter 加载 draft 模型，记录 draft attention layer 名称。
4. 根据 draft 模型是否自带 embedding/lm_head，决定是否共享 target 的 embedding/lm_head。
5. 如果启用 parallel drafting，会加载 `mask_hidden`，后续把 mask slot 的 hidden state 填成训练好的特殊向量。

### 2. target forward

本轮 target 模型处理 scheduler 安排的 token，其中可能包含上一轮的 draft token。若存在 draft token，则 target 一次 forward 会覆盖：

```text
真实 next token + draft token + bonus logits 位置
```

如果启用 EAGLE3 aux hidden states，target 返回：

```text
hidden_states, aux_hidden_states
```

runner 在 `propose_draft_token_ids()` 中根据本轮采样结果准备 drafter 输入。

### 3. 准备 EAGLE3 drafter 输入

EAGLE3 需要知道最新被接受的 token 作为下一次 draft 的起点。vLLM 分两种模式：

普通 CPU/list 路径：

- `prepare_next_token_ids_cpu()` 从真实 sampled token list 中取最后一个 token。
- 如果 chunked prefill 没有 sampled token，则从 request state 里取当前位置 token。

padded GPU 路径：

- `prepare_next_token_ids_padded()` 直接处理 GPU 上 `[num_reqs, num_spec_tokens + 1]` 的 sampled token tensor。
- rejected token 是 `-1`。
- kernel 计算每个请求的 valid token 数和 next token。
- 同时得到 `valid_sampled_tokens_count`，后续用于计算上一轮 rejected token 数。

如果本轮本来就有 speculative token，`prepare_inputs_padded()` 会根据 `valid_sampled_tokens_count` 计算：

```text
num_rejected_tokens = num_draft_tokens + 1 - valid_count
token_indices_to_sample = 当前请求最后一个有效 token 的位置
```

这样 drafter 不需要 CPU 同步就能知道每个请求从哪里继续。

### 4. EAGLE3 propose

`EagleProposer.propose()` 的主流程：

1. 如果是 `eagle3`，先调用 draft 模型的 `combine_hidden_states()`：

   ```text
   concat(aux_hidden_states) -> fc -> draft_hidden_size
   ```

2. `set_inputs_first_pass()` 构造 drafter 第一次 forward 的 input ids、position、hidden states。
3. 构造 draft attention metadata。
4. 运行 draft model。
5. 从 `token_indices_to_sample` 位置取 last hidden states。
6. 通过 lm_head greedy 取第 1 个 draft token。
7. 如果 `num_speculative_tokens == 1` 或 parallel drafting，直接返回。
8. 否则进入循环，每一步：
   - 上一步 draft token 作为下一步 input id。
   - 更新 position、seq_lens、slot_mapping。
   - 运行 draft model。
   - 取 logits argmax 得到下一个 draft token。

输出形状：

```text
[batch_size, num_speculative_tokens]
```

这批 draft token 会被复制到 CPU 或保留在 GPU，供 scheduler 下一轮调度。

### 5. 性能设计

EAGLE3 路径里有几个明显的性能取舍：

- 使用预分配 buffer，避免每 step 重复分配。
- padded drafter batch 让 GPU 路径可以避免 CPU 同步。
- 对常用输入整理逻辑写 Triton fused kernel，例如：
  - `eagle_prepare_next_token_padded_kernel`
  - `eagle_prepare_inputs_padded_kernel`
  - `eagle_step_slot_mapping_metadata_kernel`
  - `copy_and_expand_eagle_inputs_kernel`
- draft decode 阶段支持 CUDA graph，降低小 batch 小模型的 launch 开销。
- `use_local_argmax_reduction` 可避免 TP 下 all-gather 完整 vocab logits，只做局部 argmax 归约。

## DFlash 的设计理念

DFlash 在配置上属于 EAGLE family，但 runtime contract 与 EAGLE3 明显不同。

EAGLE3 是“target hidden states + token embedding 作为 draft model 输入，然后按自回归或 parallel drafting 生成 token”。DFlash 则是：

```text
target hidden states 作为 context
  -> 预先投影成每层 K/V，写入 DFlash draft KV cache
bonus token + mask token 作为 query
  -> 一次非因果 attention
  -> 并行得到多个 speculative token
```

也就是说，DFlash 把 target hidden states 变成 draft 模型的 cross-attention context。query 只包含：

```text
[next_token, mask, mask, ..., mask]
```

其中 `next_token` 是 bonus 起点，后面的 mask token 对应需要预测的 speculative positions。

因此 DFlash 强制 `parallel_drafting = True`，并且不使用 EAGLE3 的 `mask_hidden`。DFlash 使用配置里的 `dflash_config.mask_token_id` 作为 mask token id，再通过 embedding 得到 mask query。

## DFlash 执行流程

### 1. 初始化

`method == "dflash"` 时：

1. `SpeculativeConfig` 强制 `parallel_drafting = True`。
2. `GPUModelRunner` 创建 `DFlashProposer`，并设置 `use_aux_hidden_state_outputs = True`。
3. `DFlashProposer` 继承 `SpecDecodeBaseProposer`，但重写多处 EAGLE 默认行为。
4. `DFlashQwen3ForCausalLM` 加载后调用 `_build_fused_kv_buffers()`，把所有 DFlash attention 层的 KV projection 权重、K norm 权重、RoPE 参数整理成 fused buffer。

### 2. target hidden states 合并

DFlash draft 模型也有 `combine_hidden_states()`：

```text
concat(aux_hidden_states) -> fc -> draft hidden size
```

如果 `dflash_config.use_aux_hidden_state = false`，则直接使用最后层 hidden states。

`DFlashProposer._get_eagle3_use_aux_hidden_state_from_config()` 默认返回 true，也支持从 `dflash_config` 读取 `use_aux_hidden_state`。

### 3. DFlash set_inputs_first_pass

DFlash 的 `set_inputs_first_pass()` 是核心差异点。它不会把全部 context token 当作 query 跑 draft model，而是拆成两类 buffer：

- context：
  - token 数等于本轮 target token 数。
  - hidden states 保存为 `_dflash_hidden_states` 引用，不额外 copy。
  - positions 写入 `_context_positions_buffer`。
  - slot mapping 写入 `_context_slot_mapping_buffer`。
- query：
  - 每个请求固定 `1 + num_speculative_tokens` 个 query。
  - input ids 是 `[next_token, mask, mask, ...]`。
  - positions 从请求最后有效 position 后继续递增。
  - slot mapping 写入普通 `_slot_mapping_buffer`。
  - `token_indices_to_sample` 只指向 mask token，跳过 offset 0 的 bonus token。

返回的新 `CommonAttentionMetadata` 具有 DFlash 专属性质：

```text
num_actual_tokens = batch_size * (1 + num_speculative_tokens)
max_query_len = 1 + num_speculative_tokens
causal = False
seq_lens = effective_context_len + num_query_per_req
```

`causal=False` 很重要，因为 query token 需要以 cross-attention 方式看 context，并且多个 mask query 并行产生不同位置的预测。

### 4. 预写 context K/V

DFlash 的 `build_model_inputs_first_pass()` 会先调用：

```python
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,
    self._context_positions_buffer[:num_context],
    self._context_slot_mapping_buffer[:num_context],
)
```

`DFlashQwen3Model.precompute_and_store_context_kv()` 内部做：

1. 对 context hidden states 做 RMSNorm。
2. 用一个 fused GEMM 一次性算出所有层的 K/V。
3. 重新排布成 `[2, num_layers, num_ctx, num_kv_heads, head_dim]`。
4. 对每层 K 做 RMSNorm。
5. 对所有层 K 做 fused RoPE。
6. 对每层 attention 调用 `do_kv_cache_update()`，把 context K/V 写入 KV cache。

这样后续 DFlash draft forward 只需要处理 query token。

### 5. DFlash forward 与采样

context K/V 写好后，DFlash draft model 只对 query token forward：

```text
input_ids = [next_token, mask, mask, ...]
positions = query positions
attention metadata = non-causal
```

`DFlashQwen3Attention.forward()` 只计算 query 的 Q/K/V，其中 context K/V 已经在 KV cache 中。attention backend 需要支持 `causal=False`，否则 `DFlashProposer.build_per_group_and_layer_attn_metadata()` 会报错。

draft model 返回 query hidden states 后，proposer 只在 mask token 的位置取 logits：

```text
sample_hidden_states = last_hidden_states[token_indices_to_sample]
draft_token_ids = argmax(sample_hidden_states)
return [batch_size, num_speculative_tokens]
```

由于 DFlash 一次 forward 产出所有 speculative token，因此没有 EAGLE3 那种多步自回归 draft 循环。

## EAGLE3 与 DFlash 的关键差异

| 维度 | EAGLE3 | DFlash |
| --- | --- | --- |
| 配置方法 | `method="eagle3"` | `method="dflash"` |
| vLLM 分类 | EAGLE family | EAGLE family + DFlash 特化 |
| target aux hidden states | 通常需要 | 默认需要，可配置关闭 |
| draft 输入 | token ids/embeds + target hidden states | context hidden states 预写 KV，query 是 next token + masks |
| draft 方式 | 自回归多步，或 parallel drafting | 总是 parallel drafting |
| mask 表示 | parallel drafting 时可用 `mask_hidden` | 用 `dflash_config.mask_token_id` 的 embedding |
| attention | 通常 causal | 必须支持 non-causal |
| context K/V | draft forward 正常生成 | 先由 target hidden states 预计算并写入 draft KV cache |
| 输出位置 | 每步 last hidden state 采样 | 只采样 mask query 位置 |
| 实现入口 | `EagleProposer` | `DFlashProposer` |

## 对 RustInfer 的实现启发

### 1. 把 speculative decoding 设计成跨组件协议

RustInfer 不宜只在 worker 内实现一个 `draft()` 函数。vLLM 的设计说明 speculative decoding 至少需要四个协议面：

- scheduler：请求上保存 `spec_token_ids`，并把 draft token 计入下一轮待计算 token。
- worker/model runner：构造 target 验证输入、收集 logits、调用 rejection sampler、再调用 drafter。
- sampler：统一决定 accepted/recovered/bonus，不让 drafter 参与最终分布决策。
- metrics：记录 draft 数、accepted 数、每位置 acceptance rate、acceptance length。

### 2. 请求状态要区分“已验证 token”和“候选 token”

调度状态里需要类似：

```text
prompt_tokens
output_tokens
spec_token_ids
num_computed_tokens
num_output_placeholders   # async 时需要
```

被拒绝 draft token 已经消耗过一次 target forward，但不能进入真实输出，也不能永久增加 `num_computed_tokens`。

### 3. target forward 要能按 logits_indices 只计算必要 logits

验证 `k` 个 draft token 时，target 需要 `k + 1` 个 logits 位置：

- 前 `k` 个验证 draft。
- 最后 1 个是 bonus。

RustInfer 如果已有 `logits_indices` 或 last-token-only 逻辑，需要扩展成每请求多个 logits index 的 gather。

### 4. EAGLE3 需要 target 模型暴露辅助 hidden states

实现 EAGLE3 前，需要 target model 支持：

- 配置指定 aux layer ids。
- forward 返回最终 hidden state 和 aux hidden states。
- aux hidden states 顺序稳定，shape 可拼接。
- 编译/CUDA graph cache key 包含 aux layer ids。

Draft 模型侧需要：

- `combine_hidden_states()`，通常是 concat 后过 FC。
- EAGLE3 decoder 层，第一层支持 embedding + hidden state 拼接。
- 可选 vocab 映射。
- 可选共享 target embedding/lm_head。

### 5. DFlash 应作为独立 runtime contract

不要把 DFlash 只当成 `EAGLE3 + parallel_drafting`。它需要：

- context buffer 和 query buffer 分离。
- 从 target hidden states 预计算并写入 draft KV cache。
- query metadata 使用 `causal=false`。
- query input ids 固定为 `[next_token, mask...]`。
- token_indices_to_sample 只指向 mask token。
- attention backend 必须支持 non-causal decode。

RustInfer 如果当前 KV cache 写入接口只在 forward 内部发生，需要为 DFlash 增加“外部预写 K/V”的接口。

### 6. GPU 路径应避免 CPU 同步

vLLM 的 padded drafter batch 设计值得借鉴：

- sampled token 保持在 GPU tensor。
- rejected token 用 `-1` 占位。
- fused kernel 计算 valid count、next token、rejected count。
- scheduler 必须要 CPU token 时再异步 D2H copy。

这能避免每 step 为了计算 acceptance length 或 next token 强制同步。

### 7. DFlash/EAGLE3 的 max length 处理必须保守

vLLM 在 draft 输入阶段会检查：

```text
spec_decode_common_attn_metadata.max_seq_len + num_spec_tokens
<= effective_drafter_max_model_len
```

如果不满足，就给 scheduler 返回全 0 draft，避免 stale draft token 被下一轮错误使用。RustInfer 也需要类似兜底，否则接近 max model len 时容易把旧 draft token 混入新 step。

## 建议的 RustInfer 落地顺序

1. 先实现通用 speculative protocol：
   - request 上保存 `spec_token_ids`
   - scheduler 调度 draft token
   - worker 生成 `SpecDecodeMetadata`
   - rejection sampler 产生真实输出

2. 再实现 EAGLE3：
   - target aux hidden state 返回
   - EAGLE3 draft model forward
   - 自回归 draft loop
   - GPU padded path 先可后置

3. 最后实现 DFlash：
   - context/query split
   - context K/V 预写接口
   - non-causal attention metadata
   - mask-token parallel drafting

这样的顺序能先验证“lossless + scheduler 状态正确”，再优化 draft latency。

