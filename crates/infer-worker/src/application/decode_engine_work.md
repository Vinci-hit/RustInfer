# Decode Engine 重构工作记录

## 当前已完成

- `application/decode_engine.rs`
  - 新增 `DecodeEngine`。
  - `DecodeEngine` 现在拥有 `DecodeRows`。
  - decode step 主体已经搬进 `DecodeEngine::run_step`。
  - `serve_loop` 不再直接维护 decode row state，后续可以把 row state 下沉到 device tensors 而不再改 serve loop 主结构。

- `application/serve_loop.rs`
  - `decode_rows` 替换为 `decode_engine`。
  - cancel / preempt / drain 只通过 `DecodeEngine::retain_active` / `clear` 同步 row state。
  - decode step 入口改为 `decode_engine.run_step(...)`。

- `application/model_runner/cuda_decode.rs`
  - 恢复 copy-out stream 读回路径。
  - 普通 graph decode 的 `argmax_out_dev` 不再直接 `to_host_vec()`，改走 `download_argmax_out_copy_out`。
  - ABC compact 的 counts / active token / src rows / finished rows/tokens 改走 `copy_out_stream` 下载到长期 host staging。
  - 仍然保留 `synchronize_copy_out()`，所以当前只是恢复 So 路径，不是最终三路完全重叠。

- `application/forward_workspace.rs`
  - 新增 decode compact output 的长期 host staging。

## 当前仍然不是最终流水

当前 decode 仍是 host-driven：

```text
host prepare SeqStep
graph forward A -> C
merge C -> A
copy-out compact output
host commit active/decode_rows/KV
```

这保证行为接近原实现，但还没有实现真正的：

```text
copy-out(A_k) + copy-in(admissions) + forward(A_k -> C_{k+1})
wait all
merge(C_{k+1}, B, state -> A_{k+1})
```

## 最终目标

三缓冲语义：

```text
A: stable current token rows, forward/copy-out 都只读
B: new admissions / control packet
C: next sampled token from forward
```

最终 steady step：

```text
1. So: D2H copy-out A + seq_ids + finish_flags
2. Si: H2D copy-in admissions/cancels/speculative KV slots
3. Sc: CUDA graph forward reads A, writes C
4. Sc waits So + Si + graph done
5. merge kernel:
   - finished row: 不写 C 回 A，登记可回收 KV
   - active row: C -> A
   - append B admissions
   - compact rows
   - update kv_lens/generated_counts/block_tables/finish_flags
```

按这个设计，host 不需要看到 k 才能准备 k+1。可以多算一轮 KV，merge 发现上一 token finished 后回收 speculative slot。

## 下一步实现顺序

1. 删除 `worker_scheduler` 中旧的 decode 兼容代码。
   - 当前旧函数仍保留，方便无 GPU 机器上回退对照。
   - GPU 机器上验证 `DecodeEngine::run_step` 后可以删除。

2. 给 `DecodeEngine` 加长期 host scratch。
   - 复用 `order`
   - 复用 `new_indices`
   - 复用 `admit_tokens`
   - 复用 `generated_counts`
   - 复用 `max_tokens`
   - 复用 `ignore_eos`
   - 减少每 token `Vec` 分配。

3. 增加 device decode state tensors。
   - `seq_ids_dev`
   - `kv_lens_dev`
   - `generated_counts_dev`
   - `max_tokens_dev`
   - `ignore_eos_dev`
   - `finish_flags_dev`
   - `block_tables_dev`
   - `assigned_kv_dev`
   - `active_count_dev`

4. 新增 prepare/merge kernel。
   - prepare: 从 device decode state 生成 graph 所需 plan 或直接替代 plan tensors。
   - merge: 读 A/B/C/state，compact rows，更新 A/state，生成 host output packet。

5. 调整 graph capture。
   - graph 仍只包含 forward + argmax。
   - merge 保持 graph 外，符合当前三缓冲设计。

6. 真正去掉 steady decode 里的 `synchronize_copy_out()`。
   - So copy-out 只 enqueue。
   - merge 前等待 So event。
   - host output 消费可以晚一拍。

7. Pinned host memory。
   - 当前 staging 仍是 `Vec`。
   - GPU 机器上确认 bindgen 是否导出 `cudaHostAlloc/cudaFreeHost`。
   - 新增 `PinnedBuffer<T>` 后替换 copy-in/copy-out host staging。

## GPU 机器验证点

- `cargo check -p infer-worker --target-dir /tmp/rustinfer-target`
- 跑 graph priming，确认 copy-out event 没有未记录 event 报错。
- 用 `nsys` 看 steady decode：
  - Sc 是否存在 graph launch 空洞。
  - Si/So memcpy 是否真的与 Sc overlap。
  - 当前 `synchronize_copy_out()` 是否成为主要 host gap。
- 对比修改前后 token correctness。

## 风险

- 当前 copy-out 恢复后仍同步 So，性能收益有限。
- 普通 `Vec` host staging 不保证 true async overlap。
- 后续 speculative KV 会多写一轮 KV，需要在 finish 回收路径中明确归还 speculative slot。
- prefix caching 打开时，KV 回收语义要和 scheduler RadixTree 状态保持一致。
