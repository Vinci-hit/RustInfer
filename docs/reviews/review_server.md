# infer-server Crate 代码审查报告

## 组件概述

### 架构理解
`infer-server` 是 RustInfer 的 OpenAI 兼容 HTTP 推理前端，采用六边形/DDD 分层：入口层 `bin/server.rs`、路由/中间件层 `router.rs` + `middleware/request_id.rs`、API 适配层 `api/openai/*`（chat/completion/images/models + SSE 流式 + 增量 UTF-8 解码 + DTO）、领域服务层 `chat/template.rs`、出站适配器 `client/`（`InferClient` trait + `ZmqClient`：独立 ZMQ 线程 + inproc PAIR wakeup + tokio channel 桥接）、横切 `error.rs`。

### 整体质量评价
中上水平、接近生产可用：错误传播链完整、流式取消语义（`StreamHandle::Drop`）周到、UTF-8 增量解码处理了跨 token 多字节字符。主要短板：热路径 O(n²) 重复 decode、clone 偏多、两处可触发 panic 的 unwrap、DDD 依赖方向倒置、协议兼容性/边界缺陷、timeout 语义不一致与魔法数字。

### 问题数量统计
| 级别 | 数量 |
|------|------|
| Critical | 1 |
| High | 6 |
| Medium | 11 |
| Low | 8 |
| **总计** | **26** |

---

## Critical

### C1. 增量解码热路径 O(n²)：每个 token 重新 decode 整个缓冲
- **文件:行号**：`api/openai/decoder.rs:55-74`（`push`）；`flush` 同理 `:80-91`
- **类别**：性能
- **描述**：`push` 每收到一个 token 就对**整个**累积缓冲重新解码并做 char 级 prefix-delta。长度 N 的输出总成本 O(N²) 次解码 + O(N²) 字符串扫描/分配，长输出流式场景显著拉高 inter-token latency 并占用 tokio worker。
- **修复建议**：改为滑动窗口/前缀提交式增量解码（参考 vLLM `detokenize_incrementally`），维护 `prefix_offset`/`read_offset`，只 decode 尾部少量 token。

---

## High

### H1. `serde_json::to_string(...).unwrap()` 在 SSE 流热路径中可 panic
- **文件:行号**：`streaming.rs:51,85,112,114,131,149,167,171`；`:223,248,261,278,295,300`
- **类别**：逻辑/性能
- **描述**：SSE 每个 chunk `to_string(&payload).unwrap()`，模型输出含非法字符时 panic 会中断连接 task，无错误信号下发。
- **修复建议**：失败时 `tracing::error!` 并 yield `finish_reason:"error"` chunk + `[DONE]`；或用 `Event::default().json_data(&payload)`（返回 Result）。

### H2. `HeaderValue::from_str(&id).unwrap()` 可 panic（模式危险）
- **文件:行号**：`middleware/request_id.rs:22`；`:34-35` 原样回写客户端 header
- **类别**：逻辑/规范
- **描述**：自生成 UUID 不会失败，但属定时炸弹式坏味道；回写客户端 header 应考虑清洗防注入。
- **修复建议**：`expect` + 注释不变量，或直接构造已知合法 `HeaderValue`。

### H3. 非流式请求无后端响应时永久占用 pending（资源/超时语义不一致）
- **文件:行号**：`client/zmq_client.rs:431-464`（`infer`）；对照 `:283-296`、`:396-427`
- **类别**：逻辑
- **描述**：`infer`（oneshot）依赖调用方 `tokio::time::timeout`，但 `PendingRequest::Oneshot` 不带 deadline，超时清理只处理 Stream。channel 满时连 cancel 都发不出，pending 泄漏直到进程结束。
- **修复建议**：给 `Oneshot` 加 deadline，统一在 `cancel_timed_out_streams` 兜底回收。

### H4. DDD 依赖方向倒置：领域/适配层反向依赖 API DTO
- **文件:行号**：`chat/template.rs:6`（依赖 `api::openai::types::ChatMessage`）；`client/zmq_client.rs:48-54`（`StreamHandle` 被 API 层直接消费）
- **类别**：规范
- **描述**：chat 领域服务依赖最外层 DTO，违反「依赖指向内核」。
- **修复建议**：定义中立领域类型，API DTO 在 handler 转换；`StreamHandle` 抽象成端口 trait。

### H5. 流式路径每请求 `tokenizer.clone()` 全量克隆 tokenizer
- **文件:行号**：`chat.rs:117`、`completion.rs:98`→`streaming.rs:33/199`、`decoder.rs:41`
- **类别**：性能
- **描述**：每个流式请求深拷贝整个 Tokenizer（数十 MB），HF `Tokenizer::clone` 非廉价浅拷贝。
- **修复建议**：改持 `Arc<Tokenizer>`，`decode` 为 `&self` 只读共享。

### H6. ZMQ 线程内 `recv_bytes(0)` 在解析半帧时可阻塞事件循环
- **文件:行号**：`client/zmq_client.rs:249`
- **类别**：性能/逻辑
- **描述**：先 DONTWAIT 收 delimiter，再阻塞 `recv_bytes(0)` 收 payload；对端异常/截断时无限阻塞整个 ZMQ 事件循环线程，所有 in-flight 请求 stall。
- **修复建议**：payload 用带短超时接收（RCVTIMEO / 有限自旋），收不到则丢帧记录错误，绝不无界阻塞。

---

## Medium

- **M1** `zmq_client.rs:451` oneshot 超时用 `CancelReason::StreamTimeout`，语义错配，污染统计。→ 新增 `RequestTimeout`。
- **M2** `zmq_client.rs:363` 每收 chunk 硬编码重置 deadline 为 180s，覆盖配置超时（魔法数字）。→ 提 const 或复用配置 timeout。
- **M3** `chat.rs` 与 `completion.rs` 大量重复逻辑（max_tokens cap、请求构造、decode+Usage、校验）。→ 抽共享 helper。
- **M4** `chat.rs:177`/`completion.rs:146`/`streaming.rs:163,292` `u32+u32` total_tokens 可溢出（debug panic）。→ `saturating_add`/`u64`。
- **M5** `frequency_penalty`/`presence_penalty`/`seed` 声明但构造 `InferenceRequest` 时未透传，静默丢弃（`types.rs:42-46`、`chat.rs:77-90`）。→ 补齐透传或明确声明不支持。
- **M6** `chat.rs:181`/`completion.rs:150`/`images.rs:124` 非流式响应可避免的 String clone（见 M7）。
- **M7** `streaming.rs:37,40,100,103,138,141,155,157` 每 chunk `chunk_id.clone()`+`model.clone()` 累计大量小分配。→ 统一 `&str` 形式复用。
- **M8** `images.rs:48-91` n 张图串行生成，每张 clone prompt 向量。→ `try_join_all` 并发 + `Arc` 共享 token。
- **M9** `zmq_client.rs:320,357` 先试 `InferenceResponse` 失败再试 `StreamChunk`，msgpack 非 tagged 可能误判 + 重复反序列化。→ 协议层用 `#[serde(tag)]` 枚举单次分发。
- **M10** `zmq_client.rs:443-462` 与 `:322-333` timeout cancel 与 late response 竞态致冗余 cancel。→ 幂等保护。
- **M11** `images.rs:165-169` 扩散 prompt 硬编码 Qwen `<|im_start|>` 模板，未按 model_type 选择；`.take(512)` 静默截断。→ 按文本编码器类型选模板 + 超长告警。

---

## Low

- **L1** `error.rs:96-102` blanket `From<Into<anyhow::Error>>` 把所有错误降为 500，tokenize 失败本应 4xx；`ModelNotFound`/`TooManyRequests` 死代码。
- **L2** `error.rs:86-87` `OpenAIError.param`/`code` 永远 None，协议不完整。
- **L3** `api/metrics.rs:15` uptime 恒为 0，metrics 近空壳。
- **L4** `api/health.rs:11-14` `/ready` 恒 ready，不探测 scheduler 连通性。
- **L5** `completion.rs:154-173` 未校验空 prompt/空 token 数组。
- **L6** `api/openai/models.rs:15` `created:0` 非合理 epoch。
- **L7** 魔法数字散落（`zmq_client.rs:20-23,363`、`streaming.rs:183,312`、`images.rs:32,69`、`types.rs:307`）。
- **L8** `zmq_client.rs:21,342,364-368` `STREAM_CHUNK_BUFFER=64` 偏小 + `try_send` 满即 cancel，慢客户端被误杀（背压升级为取消过激进）。

---

## 总结
共审查 18 个 `.rs` 文件，发现 **26 个问题**：Critical 1 / High 6 / Medium 11 / Low 8。
最优先：**C1（增量解码改滑动窗口）→ H5（tokenizer 改 Arc）→ H1（SSE 去 unwrap）→ H3/H6（ZMQ 线程健壮性与 pending 回收）**。
