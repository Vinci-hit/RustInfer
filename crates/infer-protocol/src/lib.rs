//! # RustInfer Protocol
//!
//! RustInfer 的核心通信协议定义，包含 Server-Scheduler-Worker 三层架构的所有消息类型。
//!
//! ## 架构概览
//!
//! ```text
//! ┌─────────────┐                  ┌──────────────┐                 ┌──────────────┐
//! │   Client    │                  │    Server    │                 │  Scheduler   │
//! │  (HTTP/SSE) │                  │ (Tokenizer)  │                 │ (Orchestrator)│
//! └──────┬──────┘                  └──────┬───────┘                 └──────┬───────┘
//!        │                                │                                │
//!        │  POST /v1/completions          │                                │
//!        ├───────────────────────────────>│                                │
//!        │  { prompt: "Hello..." }        │                                │
//!        │                                │                                │
//!        │                                │  SchedulerCommand::AddRequest  │
//!        │                                ├───────────────────────────────>│
//!        │                                │  { token_ids: [123, 456] }     │
//!        │                                │                                │
//!        │                                │<───────────────────────────────┤
//!        │                                │  SchedulerOutput::Step         │
//!        │<───────────────────────────────┤  { new_token_id: 789 }         │
//!        │  data: "world"                 │                                │
//!        │                                │                                │
//!        │                                │<───────────────────────────────┤
//!        │                                │  SchedulerOutput::Finish       │
//!        │<───────────────────────────────┤  { reason: Stop }              │
//!        │  data: [DONE]                  │                                │
//! ```
//!
//! ## 协议分层
//!
//! ### 1. Frontend Protocol (Server <-> Scheduler)
//!
//! - **传输方式**: ZeroMQ PUSH/PULL
//! - **序列化**: MessagePack (高效二进制格式)
//! - **数据类型**: Token IDs (u32)，不传输原始文本
//!
//! **Server 发送命令**:
//! ```rust
//! use infer_protocol::{SchedulerCommand, InferRequest, SamplingParams, StoppingCriteria};
//!
//! let request = InferRequest {
//!     request_id: uuid::Uuid::new_v4().to_string(),
//!     input_token_ids: vec![123, 456, 789],  // Server 已完成 tokenize
//!     sampling_params: SamplingParams {
//!         temperature: 0.7,
//!         top_p: 0.9,
//!         top_k: 50,
//!         repetition_penalty: 1.1,
//!         frequency_penalty: 0.0,
//!         seed: Some(42),
//!     },
//!     stopping_criteria: StoppingCriteria {
//!         max_new_tokens: 512,
//!         stop_token_ids: vec![128001],  // EOS
//!         ignore_eos: false,
//!     },
//!     adapter_id: None,
//! };
//!
//! let cmd = SchedulerCommand::AddRequest(request);
//! // 通过 ZeroMQ 发送: socket.send(&rmp_serde::to_vec(&cmd)?)?;
//! ```
//!
//! **Scheduler 返回结果** (流式):
//! ```rust
//! use infer_protocol::{SchedulerOutput, StepOutput, FinishOutput, FinishReason};
//!
//! // Step 1: 生成第一个 Token
//! let output1 = SchedulerOutput::Step(StepOutput {
//!     request_id: "req-123".to_string(),
//!     new_token_id: 1024,
//!     logprob: Some(-0.5),
//! });
//!
//! // Step 2: 生成第二个 Token
//! let output2 = SchedulerOutput::Step(StepOutput {
//!     request_id: "req-123".to_string(),
//!     new_token_id: 2048,
//!     logprob: Some(-0.8),
//! });
//!
//! // Step 3: 完成
//! let finish = SchedulerOutput::Finish(FinishOutput {
//!     request_id: "req-123".to_string(),
//!     reason: FinishReason::Stop,
//! });
//! ```
//!
//! ### 2. Backend Protocol (Scheduler <-> Worker)
//!
//! - **传输方式**: Actor 消息传递 (例如 Tokio channels 或 ZeroMQ)
//! - **生命周期**: Worker 注册 -> 加载模型 -> 初始化 KVCache -> 执行推理
//!
//! **Worker 初始化流程**:
//! ```rust
//! use infer_protocol::{
//!     WorkerCommand, WorkerResponse, ModelLoadParams, InitKVCacheParams,
//! };
//!
//! // 1. Worker 注册
//! // let register_ack = worker.handle(WorkerCommand::Register(registration)).await?;
//!
//! // 2. 加载模型
//! let load_cmd = WorkerCommand::LoadModel(ModelLoadParams {
//!     device_id: 0,
//!     model_path: "/models/llama3-8b".to_string(),
//!     dtype: "bf16".to_string(),
//!     tp_rank: 0,
//!     tp_world_size: 1,
//!     pp_rank: 0,
//!     pp_world_size: 1,
//!     tokenizer_path: None,
//!     enable_flash_attn: true,
//!     custom_config: None,
//! });
//!
//! // 3. 初始化 KVCache
//! let cache_cmd = WorkerCommand::InitKVCache(InitKVCacheParams {
//!     num_blocks: 1024,
//!     block_size: 16,
//!     num_layers: 32,
//!     num_heads: 32,
//!     head_dim: 128,
//!     dtype: "bf16".to_string(),
//!     use_unified_memory_pool: true,
//! });
//! ```
//!
//! ## 核心设计原则
//!
//! 1. **职责分离**
//!    - Server: 处理 HTTP/文本/用户交互
//!    - Scheduler: 调度推理请求/管理 KVCache/负载均衡
//!    - Worker: 执行模型推理/GPU 计算
//!
//! 2. **纯 Token ID 传输**
//!    - Server 和 Scheduler 之间只传输 `u32` Token IDs
//!    - 避免在高并发下传输大量文本数据
//!
//! 3. **流式设计**
//!    - `SchedulerOutput::Step` 轻量级设计（仅 request_id + token_id）
//!    - 支持低延迟的 Server-Sent Events (SSE) 响应
//!
//! 4. **错误隔离**
//!    - `SchedulerOutput::Error` 独立于 `FinishReason`
//!    - 清晰区分正常结束和异常错误
//!
//! 5. **可扩展性**
//!    - 预留 `adapter_id` 支持多 LoRA 推理
//!    - 预留 `images` 支持多模态输入
//!    - 预留 `logprob` 支持 OpenAI 兼容的概率返回
//!
//! ## 序列化格式
//!
//! - **Frontend**: MessagePack (高效、紧凑)
//! - **Backend**: MessagePack 或 JSON (可配置)
//!
//! ## 版本兼容性
//!
//! 协议版本: `1.0.0`
//!
//! 当协议发生不兼容变更时，请遵循语义化版本规范：
//! - 添加可选字段: Patch 版本 (+0.0.1)
//! - 添加新的枚举变体: Minor 版本 (+0.1.0)
//! - 修改已有字段类型/删除字段: Major 版本 (+1.0.0)

use serde::{Deserialize, Serialize};
// ==================== Metrics Data Structures ====================

/// Scheduler和缓存的完整指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Scheduler统计
    pub scheduler: SchedulerStats,
    /// 缓存统计
    pub cache: CacheStats,
}

/// Scheduler统计指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// 总请求数
    pub total_requests: u64,
    /// 完成的请求数
    pub completed_requests: u64,
    /// 失败的请求数
    pub failed_requests: u64,
    /// 生成的总token数
    pub total_tokens_generated: u64,
    /// 平均排队时间 (ms)
    pub avg_queue_time_ms: f64,
    /// 平均prefill时间 (ms)
    pub avg_prefill_time_ms: f64,
    /// 平均decode时间 (ms)
    pub avg_decode_time_ms: f64,
    /// 当前队列大小
    pub queue_size: usize,
    /// 队列容量
    pub queue_capacity: usize,
    /// 当前并发请求数
    pub concurrent_requests: usize,
}

/// 缓存统计指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// 缓存命中率 (0.0 - 1.0)
    pub hit_rate: f64,
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
    /// 可驱逐的缓存大小 (tokens)
    pub evictable_size: usize,
    /// 受保护的缓存大小 (tokens)
    pub protected_size: usize,
    /// 总缓存大小 (tokens)
    pub total_cached: usize,
    /// 总缓存容量 (tokens)
    pub total_capacity: usize,
    /// 驱逐次数
    pub evictions: u64,
    /// 节点数量
    pub node_count: usize,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    /// Prefill阶段耗时 (ms)
    pub prefill_ms: u64,

    /// Decode阶段耗时 (ms)
    pub decode_ms: u64,

    /// 排队等待时间 (ms)
    pub queue_ms: u64,

    /// 实际batch大小
    pub batch_size: usize,

    /// 吞吐量 (tokens/s)
    pub tokens_per_second: f64,

    /// Decode迭代次数
    pub decode_iterations: usize,
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self {
            prefill_ms: 0,
            decode_ms: 0,
            queue_ms: 0,
            batch_size: 1,
            tokens_per_second: 0.0,
            decode_iterations: 0,
        }
    }
}

// ==================== Server <-> Scheduler Protocol ====================
//
// 这是 RustInfer 的核心入口协议，定义了 Server 和 Scheduler 之间的通信契约。
//
// ## 架构原则
// - **Server 职责**: 处理 HTTP 请求、Tokenize/Detokenize、流式响应
// - **Scheduler 职责**: 调度推理请求、管理 KVCache、分配 Worker 资源
// - **传输内容**: 仅传输 Token IDs (u32)，不传输原始文本
// - **通信方式**: ZeroMQ PUSH/PULL 模式
//
// ## 数据流
// ```
// [Client] --HTTP--> [Server] --SchedulerCommand--> [Scheduler]
//    ^                  ^            |                   |
//    |                  |            v                   v
//    |                  +---- SchedulerOutput ------[Worker Pool]
//    |                       (Streaming)
//    +-------- SSE Response <-------+
// ```

/// Server -> Scheduler 的指令
///
/// Server 通过 ZeroMQ PUSH socket 向 Scheduler 发送命令。
/// Scheduler 通过 PULL socket 接收并处理这些命令。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerCommand {
    /// 新增推理请求
    ///
    /// Server 收到 HTTP 请求后，完成 Tokenize 操作，
    /// 将 Token IDs 打包成 InferRequest 发送给 Scheduler。
    AddRequest(InferRequest),

    /// 取消推理请求
    ///
    /// 当 HTTP 客户端断开连接（或主动取消）时，
    /// Server 发送 AbortRequest 通知 Scheduler 释放资源。
    AbortRequest(String), // request_id
}

/// Scheduler -> Server 的输出
///
/// Scheduler 通过 ZeroMQ PUSH socket 向 Server 流式返回结果。
/// Server 通过 PULL socket 接收并转发给 HTTP 客户端（SSE 格式）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerOutput {
    /// 中间生成步骤 (Streaming)
    ///
    /// 每生成一个新 Token，Scheduler 就发送一个 StepOutput。
    /// Server 收到后调用 Tokenizer.decode_stream() 转换为文本，
    /// 然后通过 SSE 推送给客户端。
    Step(StepOutput),

    /// 请求完成
    ///
    /// 推理结束（遇到 EOS / 达到最大长度 / 被取消）时发送。
    /// Server 收到后关闭 SSE 连接。
    Finish(FinishOutput),

    /// 发生错误
    ///
    /// 推理过程中出现异常（如 OOM、Prompt 过长）时发送。
    /// Server 收到后返回 HTTP 错误码给客户端。
    Error(ErrorOutput),
}

// ==================== Request Structures ====================

/// 推理请求
///
/// 这是 Scheduler 处理的核心数据结构。
/// Server 已经完成了文本到 Token IDs 的转换，
/// Scheduler 只需要关注调度和生成逻辑。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    /// 唯一请求 ID (UUID v4)
    ///
    /// 用于追踪整个推理生命周期，Server 和 Scheduler 通过此 ID 关联请求。
    pub request_id: String,

    /// 输入 Token IDs
    ///
    /// **重要**: Server 已完成 Tokenize 操作 (Tokenizer.encode)，
    /// Scheduler 直接使用这些 IDs 进行推理，不需要原始文本。
    ///
    /// 类型为 u32，因为 Token ID 永远是非负整数。
    pub input_token_ids: Vec<u32>,

    /// 采样参数
    ///
    /// 虽然实际采样在 Worker 执行，但参数需要由 Server 传入。
    /// Scheduler 会在生成 WorkerInput 时透传这些参数。
    ///
    /// Scheduler 也会使用部分参数（如 max_new_tokens）来估算显存占用。
    pub sampling_params: SamplingParams,

    /// 停止条件
    ///
    /// 定义何时结束生成（EOS Token / 最大长度）。
    pub stopping_criteria: StoppingCriteria,

    /// LoRA Adapter ID (扩展字段)
    ///
    /// 预留给多 LoRA 推理场景，目前可以为 None。
    pub adapter_id: Option<String>,

    // 未来扩展:
    // pub images: Option<Vec<Tensor>>,  // 多模态图像 embeddings
    // pub prefix_cache_id: Option<String>, // System Prompt 缓存 ID
}

/// 采样参数
///
/// 控制 Token 生成的随机性和质量。
/// 这些参数最终会传递给 Worker 的 Sampler。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// 温度系数 (0.0 ~ 2.0)
    ///
    /// - 0.0: Greedy Decoding（选概率最高的 Token）
    /// - 1.0: 标准采样
    /// - >1.0: 更随机，增加多样性
    ///
    /// **默认**: 1.0
    pub temperature: f32,

    /// Nucleus Sampling 阈值 (0.0 ~ 1.0)
    ///
    /// 只从累计概率达到 top_p 的 Token 集合中采样。
    /// 例如 top_p=0.9 表示只考虑占 90% 概率质量的 Token。
    ///
    /// **默认**: 1.0 (不限制)
    pub top_p: f32,

    /// Top-K Sampling
    ///
    /// 只从概率最高的前 K 个 Token 中采样。
    /// - -1: 不限制
    /// - >0: 限制候选集大小
    ///
    /// **默认**: -1 (不限制)
    pub top_k: i32,

    /// 重复惩罚 (Repetition Penalty)
    ///
    /// 惩罚已经生成过的 Token，避免重复。
    /// - 1.0: 不惩罚
    /// - >1.0: 降低重复 Token 的概率
    ///
    /// **默认**: 1.0
    pub repetition_penalty: f32,

    /// 频率惩罚 (Frequency Penalty)
    ///
    /// 根据 Token 出现频率进行惩罚，与 OpenAI API 一致。
    /// - 0.0: 不惩罚
    /// - >0.0: 惩罚高频 Token
    ///
    /// **默认**: 0.0
    pub frequency_penalty: f32,

    /// 随机种子 (用于复现)
    ///
    /// 设置后可以保证相同输入产生相同输出。
    /// None 表示使用随机种子。
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

/// 停止条件
///
/// 定义生成何时结束。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoppingCriteria {
    /// 最大新生成 Token 数
    ///
    /// 不包括输入的 Prompt 长度。
    /// 例如: Prompt 长度 10，max_new_tokens=50，则总长度最多 60。
    ///
    /// Scheduler 会使用此参数估算 KVCache 显存占用。
    pub max_new_tokens: usize,

    /// 停止 Token IDs
    ///
    /// 遇到这些 Token 就立即停止生成。
    /// 通常包含 EOS Token ID（如 LLaMA 的 128001）。
    ///
    /// **注意**: Server 已经将停止词转换为 Token IDs。
    pub stop_token_ids: Vec<u32>,

    /// 是否忽略 EOS Token
    ///
    /// - false (默认): 遇到 EOS 就停止
    /// - true: 强制生成到 max_new_tokens，即使遇到 EOS 也继续
    ///
    /// 适用于需要固定长度输出的场景（如代码补全）。
    pub ignore_eos: bool,
}

impl Default for StoppingCriteria {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            stop_token_ids: vec![],
            ignore_eos: false,
        }
    }
}

// ==================== Response Structures ====================

/// 中间步骤输出
///
/// Scheduler 每生成一个 Token 就发送一次。
/// 设计得非常轻量，减少序列化和网络开销。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    /// 请求 ID
    pub request_id: String,

    /// 新生成的 Token ID
    ///
    /// Server 收到后调用 Tokenizer.decode_stream() 转换为文本。
    pub new_token_id: u32,

    /// Token 的对数概率 (可选)
    ///
    /// 用于调试或实现 logprobs 返回（类似 OpenAI API）。
    /// None 表示不返回概率信息（节省带宽）。
    pub logprob: Option<f32>,
}

/// 完成输出
///
/// 推理结束时发送，告知 Server 原因。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishOutput {
    /// 请求 ID
    pub request_id: String,

    /// 结束原因
    pub reason: FinishReason,

    // 注意: 不包含 num_tokens，因为 Server 可以自己统计
}

/// 采样输出 (Sampler 输出)
///
/// Model forward_paged 返回 logits，Sampler 基于 SamplingParams 生成此结果。
/// 包含一个批次内所有请求的采样结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingOutput {
    /// 采样得到的 token ID，形状 [batch_size]
    pub next_token_ids: Vec<i32>,

    /// 每个请求是否已停止，形状 [batch_size]
    pub is_stopped: Vec<bool>,

    /// 停止原因，形状 [batch_size]
    pub finish_reasons: Vec<FinishReason>,
}

/// 结束原因
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    /// 遇到了停止 Token (如 EOS)
    Stop,

    /// 达到了 max_new_tokens 限制
    Length,

    /// 被用户取消 (收到 AbortRequest)
    Abort,
}

/// 错误输出
///
/// 推理过程中发生异常时发送。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorOutput {
    /// 请求 ID
    pub request_id: String,

    /// 错误消息
    ///
    /// 例如:
    /// - "Prompt too long: 8192 tokens exceeds limit 4096"
    /// - "Out of memory: cannot allocate KV cache"
    /// - "Model not loaded"
    pub message: String,
}

// ==================== Actor-Based Scheduler-Worker Protocol ====================

/// Scheduler -> Worker 的命令消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerCommand {
    /// Worker注册请求 (握手)
    Register(WorkerRegistration),

    /// 加载模型到指定设备
    LoadModel(ModelLoadParams),

    /// 探测模型显存使用情况
    Profile(ProfileParams),

    /// 初始化KVCache
    InitKVCache(InitKVCacheParams),

    /// 执行前向推理
    Forward(ForwardParams),

    /// 查询Worker状态
    GetStatus,

    /// 卸载模型并释放资源
    UnloadModel,

    /// 健康检查
    HealthCheck,
}

/// Worker -> Scheduler 的响应消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerResponse {
    /// Worker注册确认 (握手响应)
    RegisterAck(WorkerRegistrationAck),

    /// 模型加载成功
    ModelLoaded(ModelLoadedInfo),

    /// 探测完成
    ProfileCompleted(ProfileResult),

    /// KVCache初始化完成
    KVCacheInitialized(KVCacheInfo),

    /// Forward执行完成
    ForwardCompleted(SamplingOutput),

    /// Worker状态
    Status(WorkerStatus),

    /// 模型卸载完成
    ModelUnloaded,

    /// 健康检查响应
    Healthy,

    /// 错误响应
    Error(WorkerError),
}

// ==================== Model Loading ====================

/// 模型加载参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadParams {
    /// Worker持有的GPU设备ID (e.g., 0, 1, 2, 3)
    pub device_id: u32,

    /// 模型文件路径
    pub model_path: String,

    /// 数据类型: "bf16", "fp16", "fp32", "int8", "int4"
    pub dtype: String,

    /// Tensor Parallelism: 当前rank (0-based)
    pub tp_rank: u32,

    /// Tensor Parallelism: 总共的world size
    pub tp_world_size: u32,

    /// Pipeline Parallelism: 当前rank (预留, 默认0)
    pub pp_rank: u32,

    /// Pipeline Parallelism: 总共的world size (预留, 默认1)
    pub pp_world_size: u32,

    /// Tokenizer路径 (可选，如果为None则使用model_path)
    pub tokenizer_path: Option<String>,

    /// 是否启用Flash Attention
    pub enable_flash_attn: bool,

    /// 自定义配置项 (JSON格式，用于扩展)
    pub custom_config: Option<String>,
}

/// 模型加载完成信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadedInfo {
    /// Worker ID
    pub worker_id: String,

    /// 设备ID
    pub device_id: u32,

    /// 模型名称
    pub model_name: String,

    /// 模型参数量
    pub num_parameters: u64,

    /// 模型占用的显存 (bytes)
    pub memory_used: u64,

    /// TP配置
    pub tp_rank: u32,
    pub tp_world_size: u32,

    /// 加载耗时 (ms)
    pub load_time_ms: u64,
}

// ==================== Profiling ====================

/// 探测参数 (参考vLLM的profiling机制)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileParams {
    /// 用于探测的batch size
    pub batch_size: usize,

    /// 用于探测的序列长度
    pub seq_len: usize,

    /// 探测的轮数 (多次取平均)
    pub num_rounds: u32,

    /// 是否包含prefill探测
    pub include_prefill: bool,

    /// 是否包含decode探测
    pub include_decode: bool,
}

/// 探测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// 单次forward峰值显存占用 (bytes)
    pub peak_memory_forward: u64,

    /// 模型权重占用的显存 (bytes)
    pub memory_model: u64,

    /// 总可用显存 (bytes)
    pub total_memory: u64,

    /// 推荐的KVCache可用显存 (bytes)
    pub available_kv_cache_memory: u64,

    /// Prefill平均耗时 (ms)
    pub avg_prefill_time_ms: f64,

    /// Decode平均耗时 (ms)
    pub avg_decode_time_ms: f64,

    /// 探测的batch size
    pub profiled_batch_size: usize,

    /// 探测的序列长度
    pub profiled_seq_len: usize,
}

// ==================== KVCache Initialization ====================

/// KVCache初始化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitKVCacheParams {
    /// 总的KVCache block数量
    pub num_blocks: usize,

    /// 每个block的大小 (token数量)
    pub block_size: usize,

    /// 层数 (number of layers)
    pub num_layers: u32,

    /// 注意力头数量
    pub num_heads: u32,

    /// 每个头的维度
    pub head_dim: u32,

    /// KVCache的数据类型: "bf16", "fp16", "fp32", "int8"
    pub dtype: String,
}

/// KVCache初始化完成信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheInfo {
    /// 成功分配的block数量
    pub allocated_blocks: usize,

    /// KVCache占用的总显存 (bytes)
    pub memory_used: u64,

    /// 单个block的大小 (bytes)
    pub bytes_per_block: u64,

    /// 总容量 (tokens)
    pub total_capacity_tokens: usize,

    /// 初始化耗时 (ms)
    pub init_time_ms: u64,
}

// ==================== Forward Inference ====================

/// Forward推理参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardParams {
    // ==================== 基础数据 (Flattened) ====================
    
    /// 所有请求的 token 拼成的一维数组。
    /// 形状: [total_batch_tokens]
    pub input_ids: Vec<i32>,

    /// 对应的位置索引，用于 RoPE。
    /// 形状: [total_batch_tokens]
    /// 必须显式传入，因为 Continuous Batching 下位置是不连续的。
    pub position_ids: Vec<i32>,

    // ==================== 调度元数据 (Metadata) ====================

    /// 告诉 Kernel 这一批里有多少个是 Decoding，多少个是 Prefill
    /// 这决定了 input_ids 的切分点：input_ids[0..num_decode_tokens] 是 Decode 的，然后len - num_decode_tokens 是 Prefill 的
    pub num_decode_tokens: usize,

    // ==================== PagedAttention 核心 (读) ====================

    /// 块表 (Block Table)。拍扁的一维数组。
    /// 形状: [batch_size, max_num_blocks_per_seq]
    /// 用于 Kernel 读取历史 KV。
    pub block_tables: Vec<u32>, 

    /// 每个请求的 Block Table 有效长度 (或者 max_blocks stride)
    /// 用于在 block_tables 中寻址: row_offset = req_idx * max_blocks_per_req
    pub max_blocks_per_req: usize,

    /// 每个请求的历史上下文长度 (Context Length)。
    /// 形状: [batch_size]
    /// 用于 Attention Mask 和判断循环边界。
    pub context_lens: Vec<u32>,

    // ==================== PagedAttention 核心 (写) ====================
    
    /// 槽位映射 (Slot Mapping)。
    /// 形状: [total_batch_tokens]
    /// 含义：当前的 input_ids[i] 计算出的 KV 应该写入到哪个物理位置？
    /// 格式通常是: physical_block_idx * block_size + offset_in_block
    pub slot_mapping: Vec<i32>,

    // ==================== 采样参数 ====================
    
    /// 采样只需要针对每个 Request 的最后一个 Token
    /// 这里只传索引：input_ids 中的哪些下标是 Request 的末尾？
    pub selected_token_indices: Vec<u32>,
    
    /// 具体的采样参数 (Temperature, TopP 等)，按 Request 顺序排列
    pub sampling_params: Vec<SamplingParams>,
}

/// Forward推理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardResult {
    // 形状: [batch_size]
    pub next_token_ids: Vec<i32>,
    
    // 形状: [batch_size]
    pub is_stopped: Vec<bool>,
}

// ==================== Worker Status ====================

/// Worker状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    /// Worker唯一ID
    pub worker_id: String,

    /// 设备ID
    pub device_id: u32,

    /// 当前状态
    pub state: WorkerState,

    /// 是否已加载模型
    pub model_loaded: bool,

    /// 是否已初始化KVCache
    pub kv_cache_initialized: bool,

    /// 显存统计
    pub memory_stats: MemoryStats,

    /// 性能统计
    pub performance_stats: PerformanceStats,

    /// TP配置
    pub tp_rank: Option<u32>,
    pub tp_world_size: Option<u32>,
}

/// Worker状态枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerState {
    /// 初始化中
    Initializing,

    /// 空闲
    Idle,

    /// 加载模型中
    LoadingModel,

    /// 探测中
    Profiling,

    /// 初始化KVCache中
    InitializingKVCache,

    /// 推理中
    Inferencing,

    /// 卸载模型中
    UnloadingModel,

    /// 错误状态
    Error,
}

/// 显存统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// 总显存 (bytes)
    pub total: u64,

    /// 已使用显存 (bytes)
    pub used: u64,

    /// 可用显存 (bytes)
    pub free: u64,

    /// 模型占用 (bytes)
    pub model_memory: u64,

    /// KVCache占用 (bytes)
    pub kv_cache_memory: u64,

    /// 激活值占用 (bytes)
    pub activation_memory: u64,
}

/// 性能统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// 总处理的请求数
    pub total_requests: u64,

    /// 总处理的token数
    pub total_tokens: u64,

    /// 平均prefill时间 (ms)
    pub avg_prefill_time_ms: f64,

    /// 平均decode时间 (ms)
    pub avg_decode_time_ms: f64,

    /// 吞吐量 (tokens/s)
    pub throughput_tokens_per_sec: f64,

    /// GPU利用率 (0.0 - 1.0)
    pub gpu_utilization: f32,
}

// ==================== Error Handling ====================

/// Worker错误类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerError {
    /// 错误码
    pub code: ErrorCode,

    /// 错误消息
    pub message: String,

    /// 详细的堆栈信息 (可选)
    pub details: Option<String>,

    /// Worker ID
    pub worker_id: String,

    /// 设备ID
    pub device_id: u32,
}

/// 错误码
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCode {
    /// 模型加载失败
    ModelLoadFailed,

    /// 显存不足
    OutOfMemory,

    /// KVCache初始化失败
    KVCacheInitFailed,

    /// Forward推理失败
    ForwardFailed,

    /// 设备错误 (CUDA错误等)
    DeviceError,

    /// 参数无效
    InvalidParams,

    /// Worker状态错误 (如在未加载模型时执行forward)
    InvalidState,

    /// 通信错误
    CommunicationError,

    /// 未知错误
    Unknown,
}

// ==================== Handshake Protocol ====================

/// Worker注册信息 (握手阶段发送)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistration {
    /// Worker唯一ID
    pub worker_id: String,

    /// Worker rank (TP分组中的序号)
    pub rank: u32,

    /// World size (TP组的总大小)
    pub world_size: u32,

    /// 设备类型: "cuda" 或 "cpu"
    pub device_type: String,

    /// 设备ID (仅CUDA有效)
    pub device_id: u32,

    /// 协议版本 (用于版本兼容性检查)
    pub protocol_version: String,
}

/// Scheduler对Worker注册的确认响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistrationAck {
    /// 注册状态: "ok" 或 "rejected"
    pub status: String,

    /// 响应消息
    pub message: String,

    /// Scheduler的协议版本
    pub scheduler_protocol_version: Option<String>,

    /// 分配的worker_id (如果Scheduler重新分配)
    pub assigned_worker_id: Option<String>,
}
