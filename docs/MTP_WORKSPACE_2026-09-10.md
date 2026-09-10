# MTP workspace 优化与实测（2026-09-10）

本次改动补齐 MTP 热路径中遗漏的启动预分配，延续已有 ForwardScratch / GdnScratch 的所有权与容量约束。

## 实现

- `MtpWorkspace` 在 proposer 创建时分配控制索引、固定 block table、双缓冲 hidden、logits、argmax workspace 和 token ID 存储。每轮将所有 draft 索引和 pending token 一起上传；后续 token 直接从 GPU argmax 输出送入 embedding，整轮结束才下载 draft ID。没有逐 draft 的临时 logits/hidden/argmax 张量或索引上传。
- Catch-up 对齐缓冲及两个 carry hidden 在启动时分配。carry 双缓冲保证未提交或失败的 chunk 不会改写已提交 hidden；CPU token/position Vec 保留容量。
- target 使用调用方预分配的 normalized hidden；验证器使用 Runtime 已有 argmax 输出与 workspace。
- GDN 快照在显式 MTP 启动时预留，计入 worker 启动内存预算。capture/restore 复用存储，在执行 stream 上依次排入 D2D 复制；去掉每个张量后的同步，保留事务提交所需同步。Qwen3.5-4B 每次快照仍须复制约 49.5 MiB，拒绝时仍须恢复并重放。
- Serving 验证请求、retained request 及其 token/position/block table Vec 复用容量。必要的 block table 数据填充保留；Tensor 的 Arc 句柄克隆不复制设备内容。
- 新增 `MathOps::copy_tensor`：CPU 默认同步复制，CUDA 使用指定 stream 的异步复制，校验形状、连续性、设备及部分重叠。源和目标的存储须存活至执行完成。

MTP 仍以 eager 执行；本次没有新增 draft CUDA graph、动态 K 或改变拒绝策略。LM head 权重读取、必要的状态备份与拒绝后 replay 成本仍然存在。本次也没有把整个 Runtime 宣称为“零分配”。

## GPU 对比

RTX 4070 Ti SUPER 16 GiB，Qwen3.5-4B BF16；3 个长文本 prompt，各重复 3 次，每次 128 个输出 token。每个模式独立启动，包含预热。桌面仍使用该 GPU，测试显式允许最多 4096 MiB 背景显存和 60% 预检利用率；因此结果是该环境下的观测，不是隔离生产环境的保证。

先跑预分配版本，再跑保留的旧二进制，最后跑完成 pending-token 合并上传的最终版本。下表使用旧版与最终版结果，总吞吐按总输出 token / 总请求耗时计算。

| 模式 | 旧版 tokens/s | 最终版 tokens/s | 变化 | Draft 接受率（旧 → 新） |
|---|---:|---:|---:|---|
| graph | 61.41 | 61.61 | +0.33% | — |
| mtp1 | 67.65 | 71.22 | +5.28% | 527/612 (86.11%) → 525/612 (85.78%) |
| mtp3 | 62.23 | 67.00 | +7.66% | 717/1248 (57.45%) → 717/1248 (57.45%) |

轮次耗时中位数（ms）；verify 包含必要的快照、恢复及 replay：

| 模式 | Draft | Verify | Catch-up |
|---|---:|---:|---:|
| mtp1 | 2.832 → 2.571 | 20.667 → 19.778 | 0.992 → 0.706 |
| mtp3 | 8.625 → 7.675 | 37.553 → 35.641 | 0.981 → 0.697 |

## 正确性与验证边界

- worker CPU 测试 141 + hybrid integration 21 项通过；覆盖分块 prefill、拒绝回滚、请求复用、索引 tile 边界、缓冲地址复用、未提交 carry 隔离和嵌套 Vec 复用。
- CPU Clippy `-D warnings`、格式检查及 CUDA release 构建通过。
- GPU 专项测试 `scoped_copy_is_capture_safe_and_rejects_partial_overlap` 通过：复制可被 graph capture，重放读取更新后的源，部分重叠返回错误。
- 新旧版所有短文本均为 3/3；长文本与各自普通 graph 解码仍不全一致，stop-string 检查在普通模式及 MTP 都失败。这些是旧版已有的正确性缺口，本次没有宣称解决。基准会完整保存结果并以失败退出，不能把它视为全绿验收。
- graph：最终版长文本与 graph 一致 9/9；与同模式旧版一致 8/9。
- mtp1：最终版长文本与 graph 一致 6/9；与同模式旧版一致 7/9。
- mtp3：最终版长文本与 graph 一致 3/9；与同模式旧版一致 9/9。

普通 graph 跨运行也可能变化；因此不将所有输出差异归因于此次内存优化，也不据此证明与普通解码严格等价。

## 复现与原始记录

```bash
CUDA_ARCH=sm_89 cargo build --release --locked -p infer-worker -p infer-scheduler -p infer-server
python3 scripts/bench_qwen35_mtp.py --model /root/models/Qwen3.5-4B --gpu 0 \
  --output target/mtp-bench-workspace-final --repeats 3 --tokens 128 \
  --modes graph mtp1 mtp3 --max-background-memory-mib 4096 --max-background-utilization 60
```

输出目录必须不存在；若已存在，请换新目录。CPU 与 CUDA 专项验证命令：

```bash
cargo test -p infer-worker --no-default-features --lib --tests
cargo clippy -p infer-worker --no-default-features --lib --tests -- -D warnings
CUDA_ARCH=sm_89 cargo test -p infer-backend-cuda --test graph_memory scoped_copy_is_capture_safe_and_rejects_partial_overlap -- --ignored --test-threads=1
```

- 旧版结果：`target/mtp-bench-workspace-before/results.json`
- 初版优化结果：`target/mtp-bench-workspace-after/results.json`
- 最终结果：`target/mtp-bench-workspace-final/results.json`
- 各目录保留配置、响应全文、worker 日志、GPU 信息及二进制 SHA256。

旧 worker SHA256：`7f331ff21d0efb8305293bda548462db78fd166fb8bbe361f3823013345fd146`

最终 worker SHA256：`9f50eff9a9f273d18a36f2f284d2590d84e9aed7682cd9295265186b221bfc81`


## 后续小优化：复用设备 token 缓冲

基于 `9099b5e`，proposer 将 pending token 与 draft ID 保留在连续的设备缓冲中。Target 验证直接读取该缓冲；拒绝后的单序列重放和 MTP catch-up 读取其保留前缀。省掉验证、重放（发生时）和 catch-up 的重复 token H2D。为拼成连续输入，新增一次 4-byte pending token D2D。

本轮仍在 CPU 上验证接受前缀，draft ID 的 D2H 和现有事务同步仍保留；没有加入 CUDA Graph、GPU 接受判断、跨轮 issue/finalize 或 GDN 前缀状态保存。该内部接口只接收同一次 proposer 输出的 host/device 配对数据，仅用于单序列 MTP。普通、多序列和 prefill 继续走原有接口。

验证：144 项 worker 单测 + 21 项 hybrid 测试通过；CPU Clippy `-D warnings`、格式检查和 CUDA release 构建通过。新增测试覆盖零 draft、不同接受前缀、EOS 截断、错误形状/多序列拒绝，并断言验证与重放不触碰 host token 上传缓冲、catch-up 元数据更新不覆盖设备 token。

同卡新旧二进制各跑 graph / K=1 / K=3，3 个 prompt × 3 次 × 128 token，配置同上。总吞吐如下：

| 模式 | 9099b5e tokens/s | 设备 token 复用 tokens/s | 变化 | 接受数/提议数（旧 → 新） |
|---|---:|---:|---:|---|
| graph | 57.90 | 57.69 | -0.37% | — |
| mtp1 | 68.76 | 68.34 | -0.61% | 528/612 → 527/612 |
| mtp3 | 63.43 | 63.68 | +0.40% | 717/1248 → 717/1248 |

该轮结果不足以证明稳定的端到端吞吐提升。省掉的 ID 上传非常小，且仍有 CPU 接受判断、状态同步及模型计算；共享桌面 GPU 的波动也会影响结果。不把本次改动宣传为已实现完整 ABC 加速。

新旧版短文本均 3/3 通过；与普通解码的长文本差异、stop-string 检查失败仍存在，基准仍以失败退出。

原始记录：`target/mtp-bench-direct-before/results.json`、`target/mtp-bench-direct-after/results.json`，含配置、响应、日志和二进制 SHA256。
