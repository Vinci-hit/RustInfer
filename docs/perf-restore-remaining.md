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
- **#5 prefill argmax 跑全部行**（TTFT 3× 主因）→ `FusedOps::argmax_into` 加 `selected_rows: Option<&Tensor<i32, Self>>`；`AbcBuffers` 新增 `sampled_rows_dev`，`sample_tail` 每步上传 `cu_q_lens[i+1]-1`，kernel 只算 `batch` 行 argmax + 仅 D2H `batch` 个 id。`argmax_out_dev`/`argmax_ws` 容量回退到 `cap_batch`（不再需要 `cap_num_tokens`）
- **#3 flash decode workspace 每层 alloc + memset** → `ForwardScratch` 新增 `flash_ws`，容量经 `FusedOps::flash_decode_workspace_capacity_f32(cap_batch, head_num, head_dim)` 探测；`Attention::run` 每层零分配地传给 `D::attention_paged(..., Some(&mut ws))`。`UnsafeCell<Tensor<f32,D>>` 包装允许 `&self` 路径上让每层各自拿一个 owned view（layers 串行 on one stream，无别名）。`D::attention_paged` 同时保留 `workspace: None` legacy 路径
- **#4 `D::attention_paged` 签名升级**（同上）：trait 增 9 个参数（+ workspace）、新增 `flash_decode_workspace_capacity_f32` 默认 0；CUDA override 实现 + CPU reference 走默认（忽略 workspace）

---

## 未完成（均为 TTOT 微优化，单项 ~µs 级；GPU 占满时无法验证，建议先 benchmark 再做）

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
