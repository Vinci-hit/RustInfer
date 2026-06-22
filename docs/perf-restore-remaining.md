# Pre-refactor 性能恢复 —— 未完成项

承接 `docs/perf-regression-buffer-pipeline.md`。基线对比：重构前 `1cd848c`（快）vs 当前。

## 已恢复（已落地，CPU forward 测试验证数据流正确）

- 每层 ~396 次 `D::alloc_tensor`（`cudaFree` 同步风暴）→ 预分配 `ForwardScratch`（`crates/infer-worker/src/domain/forward_scratch.rs`，`Runtime::new` 经 `DecoderModel::install_scratch` 注入）
- `split_qkv` 拷贝 → `FusedOps::qkv_split`：CUDA 返回 `qkv` 的零拷贝列 narrow（kernel 认 stride），CPU 默认实现物化连续副本
- 同步 `cudaMemset` → `cudaMemsetAsync` + 尺寸键回收池（`infer-backend-cuda` `config.rs`/`lib.rs`，`POOL_RETAIN_BUDGET` 上限 + 锁外 evict）
- `logits [tokens, vocab]` 每步 alloc（且撑大 pool）→ 预分配 `ForwardScratch.logits`，`Decoder::finalize` 复用
- prefill `input_ids` 每步阻塞 `cudaStreamSynchronize` → 持久 `Runtime::prefill_ids_buf` + host 镜像 + `upload_i32_prefix`（async）
- 丢失 `fused_add_rmsnorm` → deferred-delta：`Hidden.pending` 携带上一 sublayer 的残差 delta，下一 sublayer 的 pre-norm 用 `fused_add_rmsnorm` 融合 add+norm；`decode_layers` 末尾 flush。CUDA 真融合，CPU 走 default(add+rmsnorm) 行为不变
- **argmax 容量 latent bug**（`total_tokens > cap_batch` 的 prefill 必报错）→ `argmax_out_dev`/`argmax_ws` 按 `cap_num_tokens` 分配；`sample_tail` 传 `[logits_rows]` view 给 `argmax_into`

---

## 未完成（均为 CUDA 专属微优化，单项 ~µs 级；GPU 占满时无法验证，建议先 benchmark 再做）

### #3 flash-attention workspace 每层 alloc + memset（进 decode graph）
- **现状**：`crates/infer-backend-cuda/src/lib.rs:~642` `attention_paged` 内 `Tensor::<f32>::zeros([workspace_elems], q.device())` 每次调用分配 + 清零。decode 走 graph 捕获时该 memset 被录进图，每 token 每层（~36 次）重放。
- **旧版**：`forward_workspace.rs` 预分配 `flash_decode_workspace_f32` 一次（仅构造时清零），`attention_paged` 收 `workspace: &mut Tensor<f32>` 参数复用。
- **影响**：TTOT，~18µs/token（量级估计），kernel 写前不读所以 memset 本就多余。
- **修法**：(A) 给 `attention_paged` 重加 `workspace` 参数（trait `fused_ops.rs` + CUDA impl + reference + 调用方 `attention.rs`），`ForwardScratch` 增一个 f32 flash buffer 传入。难点：flash 容量公式 `flash_decode_workspace_capacity_f32` 是 CUDA 专属，而 `ForwardScratch`/`Runtime` 是 generic-over-`D`，需把容量作为参数传入（旧版 `ForwardWorkspace::new` 正是这么做）。(B) 走 uninit alloc（跳过 memset）——需动 `infer-core` 分配 API。
- **难度**：中-高（签名改动 + generic-D 容量传递）。

### #5(perf) prefill argmax 跑全部行而非每序列末行
- **现状**：`crates/infer-worker/src/application/runtime.rs` `sample_tail` 对全部 `num_tokens` 行做 argmax，host 端再挑 `offset+q_len-1`。（容量正确性已修，仅性能未优。）
- **旧版**：`argmax_batched` 用 `selected_rows = cu_q_lens[seq+1]-1`，只算 `batch` 个 argmax。
- **影响**：TTFT，长 prompt ~150µs（P 行 vs 1 行的 argmax + 更大 D2H）。注意 lm_head 仍投影全部行（新旧一致，是 `SampleRows` 预留 seam，非回归）。
- **修法**：给 `argmax_into` 重加 `selected_rows` 参数（kernel 本就支持），或先 gather 末行 logits 再 argmax。
- **难度**：中（签名改动）。

### #4 CUDA graph 不在 bootstrap 预捕获
- **现状**：`runtime.rs:~521` `prime_graphs` 只 `GraphRunner::new`，不 warmup/capture。首次命中某 batch size 的 decode 步在服务热路径上付冷捕获（一次 eager forward + host sync + capture，见 `issue_decode_abc` cold path ~944-963）。
- **旧版**：`cuda_graph_runner.rs::warmup_and_capture_all` 在进 serve loop 前对每个 capture size 预热 + 捕获 + 验证 launch + sync。
- **影响**：TTOT，每个 batch size **一次性** spike（非持续；15 个 size 各一次，之后命中缓存）。
- **修法**：`prime_graphs` 移植 warmup：对 `capture_sizes` 每个尺寸用 scratch block 跑两次 dummy decode forward，capture forward+finalize+argmax，验证 launch 一次。
- **难度**：中（worker-local，但需构造 dummy 解码状态）。

### #6 decode stop-metadata H2D 在 compute 流（非 copy-in 流）
- **现状**：`runtime.rs:~969-980` gen/max/ignore/eos 经 `upload_i32_prefix` → `upload_async` 在 `config.stream`（主 compute 流），串在 forward 与 merge 之间。
- **旧版**：经 `upload_h2d_copy_in` 在 `copy_in_stream`(Si) + `record_copy_in`，与 forward 重叠，merge 前 `compute_wait_copy_in`。
- **影响**：TTOT，~3µs/step（丢 DMA 重叠）。
- **修法**：改走 `cfg.upload_h2d_copy_in` + `record_copy_in`，merge 前 wait。
- **难度**：低-中，但 **stream ordering 改错会腐蚀 decode**，需谨慎。

### #7 每步重建 stop-metadata host Vec
- **现状**：`runtime.rs:~969` 每步 `gen_i32/max_i32/ign_i32` 三个 `Vec` collect + H2D。
- **影响**：TTOT，~100ns/step（琐碎）。
- **修法**：`AbcBuffers` 加持久 pinned i32 镜像，原地填充，仅变化时上传。
- **难度**：低。

### #8 `plan_ragged_tiles` 每 prefill 算两次
- **现状**：`build_plan` 与 `upload_index`（`runtime.rs:~712,750`）各算一次。
- **影响**：TTFT，~µs CPU（琐碎）。
- **修法**：`build_plan` 算一次存到 `BatchPlan`（`infer-core/src/plan.rs`），`upload_index` 复用。
- **难度**：低（跨 crate，动 `BatchPlan`）。

---

## 杂项
- `crates/infer-worker/src/domain/mod.rs`：`tensor_tests` 被临时 `#[cfg(any())]` 禁用 —— 它有**预存在**的 `MathOps`/`CoreOps` 歧义 + 过时 `sdpa` 签名编译错（与本次无关）。需单独修复后重新启用。
- 验证：`cargo test -p infer-worker --lib component_decoder_ragged_batch_matches_serial` 通过（CPU，ragged==serial）。其余 CUDA lib 测试在共享 GPU 满载时 flaky OOM（环境问题，非逻辑）。
- benchmark：见 `bench/bench_online_qps.py`，对比 `1cd848c`。
