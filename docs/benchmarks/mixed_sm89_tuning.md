# SM89 Mixed graph 与 admission 实验

整理日期：2026-09-22。RTX 4070 Ti SUPER，Qwen3-4B BF16。

本轮保留可复现实验实现，**五项新开关默认均不启用**；已有 Mixed graph、CuTe Attention 和 HTTP 修复继续保留。没有把扩大分桶、合入更多 prefill 或改变容量设为默认策略。

## 配置与边界

配置由 `application/mixed_tuning.rs` 的 `MixedTuning::from_env` 在 `Runtime::new` 统一解析一次；graph 预热、shape 选择和调度使用同一份配置，不在请求热路径重复读取环境变量，也不使用跨 Runtime 的全局缓存。启动后修改环境变量不会改变现有 graph。

| 环境变量 | 未设置时 | 含义与约束 |
|---|---|---|
| `RUSTINFER_MIXED_GRAPH_EXTRA_TOKENS` | 不增加桶 | 逗号分隔的额外 token 桶，例如 `520,1032,2056`；每项必须是正的 8 倍数，解析后排序去重，空字符串/空项报错。增加启动预热覆盖，受 token/序列容量、预热数量和 graph 总数限制；不保证任意 row/tile shape 都命中。它不改变 admission 预算。 |
| `RUSTINFER_MIXED_GRAPH_EXACT_DECODE` | 沿用 capture slot 前缀 | 只接受 `0`/`1`；`1` 启用按实际 decode 行数选择/预热前缀，减少前缀取整产生的额外工作；增加 graph 数量，仍受预热上限限制。 |
| `RUSTINFER_MIXED_GRAPH_SELECTED_READOUT` | 沿用原 readout | 只接受 `0`/`1`；`1` 启用 GPU gather 后做 final norm / LM head / argmax，只投影需要输出的行。索引每步更新，scratch 仅启用时分配，地址在 graph capture 前固定。 |
| `RUSTINFER_MIXED_ADMISSION_TOKENS` | 沿用既有 mixed step 预算 | 正整数，覆盖本步 admission 的总 token 软预算，包含预留的 decode capture slot；裁到运行时硬容量。不是单请求 chunk 大小，也不会自行扩大内存容量。 |
| `RUSTINFER_MIXED_MAX_PREFILLS` | 不增加行数限制 | 正整数，限制 admission 的 **prefill 行数**，不是命令数量。命令包含多行时按行计数。 |

非法配置在启动时返回错误；例如布尔开关写成 `true` 不会静默启用或忽略。Admission 按 FIFO 处理，遇到不能容纳的命令后延后其余命令。命令原子地 admission；为保证进展，第一条命令即使超过 token 或行数软上限也会放行。因此 `MAX_PREFILLS=1` 不能承诺每次只执行一条 prefill 行。后续 forward 组装仍遵守实际 token/row 硬容量，必要时拆成多个 forward。

## 保留什么，未采用什么

- **保留 selected readout 实现，显式启用。** 原 Mixed graph 对 2056 行做词表投影后才选 8 行；Nsight 定位到约 19.4 ms 的额外 GEMM。先 gather 后投影使固定 Mixed forward 从 246.78 降至 227.33 ms；16 个对照请求完整文本一致。这是已测样本证据，不等于全部模型/shape 的数值证明。
- **额外桶和 exact decode 保留为实验选项。** 只补 graph 桶没有稳定的端到端胜出；不要据此宣称补桶解决了 TPOT。
- **扩大 admission 未设为默认。** 容量 4160 的两轮正反顺序对照中，两条 prefill 的平均 TPOT 约降低 4.7%、吞吐约提高 2.1%，但 TTFT 增加约 14.9%，输出间隔 P99 从约 232 ms 增至 447 ms。不存在没有代价的胜出。

4160 是本轮实验的 `max_batch_tokens`，此前为 4096；两个 2048-token prefill 再加 decode 无法装进 4096。单请求 `chunked_prefill_size=4096`、`max_model_len=4096` 均未改变。不能把 4160 的成绩直接归因为开关变化或与旧容量的 vLLM 成绩作严格公平对照。

## 数据与验证

- [Graph 实验报告](/root/models/benchmarks/mixed-graph-tuning-20260921/REPORT.md)：分桶、decode 前缀及 selected readout 对照。
- [Nsight 逐阶段报告](/root/models/benchmarks/mixed-vllm-profile-20260921/REPORT.md)：LM head 开销、TTFT/TPOT 归因与文本一致性样本。
- [Admission 实验报告](/root/models/benchmarks/mixed-admission-20260921/REPORT.md)：保留原始请求、summary 和日志，列出实际完成的正反顺序重复实验。

完整 worker lib tests 此前被既有 `models/int4_tests.rs:228` 类型推断错误阻塞；当时编译及提取生产逻辑的局部测试通过。本次源码整理后的构建和定向测试状态单独记录，不能把局部测试表述为全套通过。

### 2026-09-22 整理后的验证

- CUDA release worker 构建通过：`cargo build --release --locked -p infer-worker`（保留既有 SM89/CuTe 构建环境）。
- `cargo test --release --locked --no-default-features -p infer-worker --lib mixed_`：13 项通过，包括配置/策略、graph shape/key/warmup 和既有 recurrent 测试。测试直接使用仓库实现，不再依赖从函数中抽取代码的临时脚本。
- `git diff --check` 通过。未启动模型或再次运行 GPU benchmark，因此以上是本次整理的构建与 CPU 回归验证；前文性能数字仍来自历史实测。

本次清理集中配置解析、移除临时 admission 日志、使 graph shape 使用显式配置，并把误导性的 `forward_finalize_argmax_all_selected` 重命名为 `forward_mixed_graph_argmax`。历史原始数据和已实现功能均保留。
