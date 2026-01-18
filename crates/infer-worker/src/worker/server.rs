//! Worker Server - ZeroMQ-based RPC Server for Worker
//!
//! This module provides an async server that exposes the Worker functionality
//! over ZeroMQ for communication with the Scheduler.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Scheduler                                     │
//! │                        (Router Socket)                                  │
//! └────────────────────────────┬────────────────────────────────────────────┘
//!                              │ ZeroMQ (IPC/TCP)
//!              ┌───────────────┼───────────────┐
//!              ▼               ▼               ▼
//! ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
//! │  WorkerServer  │ │  WorkerServer  │ │  WorkerServer  │
//! │   (rank=0)     │ │   (rank=1)     │ │   (rank=2)     │
//! │ Dealer Socket  │ │ Dealer Socket  │ │ Dealer Socket  │
//! └────────────────┘ └────────────────┘ └────────────────┘
//! ```
//!
//! # Protocol
//!
//! - **Request**: Scheduler sends `WorkerCommand` (bincode serialized)
//! - **Response**: Worker sends `WorkerResponse` (bincode serialized)
//!
//! # Usage
//!
//! ```ignore
//! use infer_worker::worker::WorkerServer;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let server = WorkerServer::new(
//!         0,                                    // rank
//!         1,                                    // world_size
//!         "ipc:///tmp/rustinfer-scheduler.ipc", // scheduler_url
//!     ).await?;
//!
//!     server.run_loop().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Architecture Flow
//!
//! 1. Worker starts with just rank/device info
//! 2. Worker connects to Scheduler
//! 3. Worker sends Register message
//! 4. Scheduler sends LoadModel command (with model path)
//! 5. Worker loads model from specified path
//! 6. Scheduler sends InitKVCache command
//! 7. Worker is ready for inference

use std::time::Instant;

use anyhow::Result;
use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

use infer_protocol::{
    WorkerCommand, WorkerResponse, WorkerError, ErrorCode,
    ModelLoadParams, ModelLoadedInfo,
    ProfileParams, ProfileResult,
    InitKVCacheParams, KVCacheInfo,
    ForwardParams, ForwardResult,
    WorkerStatus,
    WorkerState as ProtocolWorkerState,
    MemoryStats as ProtocolMemoryStats,
    PerformanceStats as ProtocolPerformanceStats,
    WorkerRegistration, WorkerRegistrationAck,
};

use crate::base::DeviceType;
use crate::base::DataType;
use crate::model::ModelFactory;
use crate::tensor::Tensor;

use super::{Worker, WorkerConfig};
use super::worker::WorkerState;

/// Helper function to create an i32 tensor from a slice
fn create_i32_tensor(data: &[i32], device: DeviceType) -> Result<Tensor> {
    let len = data.len();

    // Create CPU tensor and fill with data
    let mut cpu_tensor = Tensor::new(&[len], DataType::I32, DeviceType::Cpu)?;
    cpu_tensor.as_i32_mut()?.as_slice_mut()?.copy_from_slice(data);

    // Move to target device if needed
    match device {
        DeviceType::Cpu => Ok(cpu_tensor),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(device_id) => cpu_tensor.to_cuda(device_id),
    }
}

/// Worker Server - Async RPC server for Worker
///
/// Manages the Worker instance and handles ZeroMQ communication
/// with the Scheduler.
pub struct WorkerServer {
    /// The underlying Worker instance
    worker: Worker,

    /// ZeroMQ Dealer socket for communication
    socket: zeromq::DealerSocket,

    /// Worker rank (for TP/PP)
    rank: usize,

    /// Total world size
    world_size: usize,

    /// Whether server is running
    running: bool,
}

impl WorkerServer {
    /// Create a new WorkerServer
    ///
    /// # Arguments
    /// * `rank` - Worker rank (0-indexed), maps to GPU device
    /// * `world_size` - Total number of workers
    /// * `scheduler_url` - ZeroMQ endpoint for Scheduler (e.g., "ipc:///tmp/scheduler.ipc")
    ///
    /// # Returns
    /// A new WorkerServer ready to run
    ///
    /// # Note
    /// The model path is NOT provided here. The Scheduler will send
    /// a LoadModel command with the model path to load.
    pub async fn new(
        rank: usize,
        world_size: usize,
        scheduler_url: &str,
    ) -> Result<Self> {
        // Create Worker with appropriate device type
        #[cfg(feature = "cuda")]
        let worker_config = WorkerConfig::cuda(rank as u32)
            .with_id(format!("worker-{}", rank))
            .with_tp(rank as u32, world_size as u32);

        #[cfg(not(feature = "cuda"))]
        let worker_config = WorkerConfig::cpu()
            .with_id(format!("worker-{}", rank))
            .with_tp(rank as u32, world_size as u32);

        let worker = Worker::new(worker_config)?;

        // Initialize ZeroMQ Dealer socket
        let mut socket = zeromq::DealerSocket::new();

        // Connect to Scheduler
        println!("[Worker-{}] Connecting to scheduler at {}", rank, scheduler_url);
        socket.connect(scheduler_url).await?;
        println!("[Worker-{}] Connected successfully", rank);

        Ok(Self {
            worker,
            socket,
            rank,
            world_size,
            running: false,
        })
    }

    /// Run the main event loop
    ///
    /// This is a blocking loop that:
    /// 1. Performs handshake with Scheduler
    /// 2. Receives commands from Scheduler
    /// 3. Dispatches to appropriate handler
    /// 4. Sends response back
    ///
    /// The loop runs until an error occurs or shutdown is requested.
    pub async fn run_loop(&mut self) -> Result<()> {
        println!("[Worker-{}] Starting main loop...", self.rank);
        self.running = true;

        // 【握手阶段】在开始处理任务前先与 Scheduler 建立连接
        if !self.perform_handshake().await? {
            return Err(anyhow::anyhow!("Handshake with Scheduler failed"));
        }

        println!("[Worker-{}] Handshake successful, ready to accept tasks", self.rank);

        while self.running {
            // 1. Receive message from Scheduler
            let msg: ZmqMessage = match self.socket.recv().await {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[Worker-{}] ZMQ recv error: {}", self.rank, e);
                    continue; // Network hiccup, retry
                }
            };

            // 2. Extract frames from DealerSocket message
            // DealerSocket 格式：[empty_delimiter_frame, payload_frame]
            let frames = msg.into_vec();
            if frames.len() < 2 {
                eprintln!("[Worker-{}] Invalid message format: expected 2 frames, got {}", self.rank, frames.len());
                continue;
            }

            // frames[0] 是空的分隔符帧（来自 Scheduler）
            // frames[1] 是实际的命令载荷
            let payload = &frames[1];

            // 3. Deserialize command
            let command: WorkerCommand = match bincode::deserialize(payload) {
                Ok(cmd) => cmd,
                Err(e) => {
                    eprintln!("[Worker-{}] Deserialize error: {}", self.rank, e);
                    let error_response = self.make_error(
                        ErrorCode::CommunicationError,
                        format!("Failed to deserialize command: {}", e),
                    );
                    self.send_response(error_response).await?;
                    continue;
                }
            };

            // 4. Handle command (synchronous - GPU is exclusive resource)
            let response = self.handle_command(command).await;

            // 5. Send response back (with error handling - don't exit on send failure)
            if let Err(e) = self.send_response(response).await {
                eprintln!("[Worker-{}] Failed to send response: {}", self.rank, e);
                // Continue running even if send fails
            }
        }

        println!("[Worker-{}] Main loop exited", self.rank);
        Ok(())
    }

    /// Stop the server gracefully
    pub fn shutdown(&mut self) {
        println!("[Worker-{}] Shutdown requested", self.rank);
        self.running = false;
    }

    /// Perform handshake with Scheduler
    ///
    /// This method:
    /// 1. Sends WorkerRegistration to Scheduler
    /// 2. Waits for RegisterAck response (with timeout)
    /// 3. Validates the response
    ///
    /// Returns Ok(true) if handshake successful, Ok(false) if rejected
    async fn perform_handshake(&mut self) -> Result<bool> {
        println!("[Worker-{}] Performing handshake with Scheduler...", self.rank);

        // 准备注册信息
        let device_type_str = match self.worker.device_type() {
            DeviceType::Cpu => "cpu".to_string(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => "cuda".to_string(),
        };

        let device_id = match self.worker.device_type() {
            DeviceType::Cpu => 0,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => id as u32,
        };

        let registration = WorkerRegistration {
            worker_id: self.worker.worker_id().to_string(),
            rank: self.rank as u32,
            world_size: self.world_size as u32,
            device_type: device_type_str,
            device_id,
            protocol_version: "0.1.0".to_string(),
        };

        // 发送注册请求
        let register_cmd = WorkerCommand::Register(registration.clone());
        let payload = bincode::serialize(&register_cmd)?;

        // DealerSocket 发送格式：[empty_frame, payload]
        // RouterSocket 接收格式：[address, empty_frame, payload]
        let mut msg = ZmqMessage::try_from(Vec::<u8>::new())?; // empty delimiter frame
        msg.push_back(payload.into());
        self.socket.send(msg).await?;
        println!(
            "[Worker-{}] Sent registration: worker_id={}, rank={}, device={}:{}",
            self.rank,
            registration.worker_id,
            registration.rank,
            registration.device_type,
            registration.device_id
        );

        // 等待 ACK 响应（带超时）
        let handshake_timeout = tokio::time::Duration::from_secs(10);
        let recv_result = tokio::time::timeout(handshake_timeout, self.socket.recv()).await;

        match recv_result {
            Ok(Ok(msg)) => {
                // 成功接收到消息
                let payload = msg.into_vec().pop().ok_or_else(|| {
                    anyhow::anyhow!("Empty ACK message received during handshake")
                })?;

                let response: WorkerResponse = bincode::deserialize(&payload)?;

                match response {
                    WorkerResponse::RegisterAck(ack) => {
                        if ack.status == "ok" {
                            println!(
                                "[Worker-{}] ✓ Handshake successful: {}",
                                self.rank, ack.message
                            );
                            if let Some(version) = ack.scheduler_protocol_version {
                                println!("[Worker-{}]   Scheduler protocol version: {}", self.rank, version);
                            }
                            Ok(true)
                        } else {
                            eprintln!(
                                "[Worker-{}] ✗ Handshake rejected: {}",
                                self.rank, ack.message
                            );
                            Ok(false)
                        }
                    }
                    _ => {
                        eprintln!(
                            "[Worker-{}] ✗ Unexpected response during handshake: {:?}",
                            self.rank, response
                        );
                        Ok(false)
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("[Worker-{}] ✗ ZMQ error during handshake: {}", self.rank, e);
                Err(anyhow::anyhow!("ZMQ error during handshake: {}", e))
            }
            Err(_) => {
                eprintln!("[Worker-{}] ✗ Handshake timeout (10s)", self.rank);
                Err(anyhow::anyhow!("Handshake timeout after 10 seconds"))
            }
        }
    }

    /// Handle incoming command and return response
    async fn handle_command(&mut self, cmd: WorkerCommand) -> WorkerResponse {
        match cmd {
            WorkerCommand::Register(registration) => self.handle_register(registration),
            WorkerCommand::LoadModel(params) => self.handle_load_model(params),
            WorkerCommand::Profile(params) => self.handle_profile(params),
            WorkerCommand::InitKVCache(params) => self.handle_init_kv_cache(params),
            WorkerCommand::Forward(params) => self.handle_forward(params),
            WorkerCommand::GetStatus => self.handle_get_status(),
            WorkerCommand::UnloadModel => self.handle_unload_model(),
            WorkerCommand::HealthCheck => WorkerResponse::Healthy,
        }
    }

    /// Handle Register command (for protocol completeness)
    ///
    /// Note: In current design, Worker initiates handshake.
    /// This handler is for future compatibility if Scheduler needs to re-register.
    fn handle_register(&self, registration: WorkerRegistration) -> WorkerResponse {
        println!(
            "[Worker-{}] Received registration request from Scheduler",
            self.rank
        );

        // Validate registration
        if registration.rank != self.rank as u32 {
            return WorkerResponse::RegisterAck(WorkerRegistrationAck {
                status: "rejected".to_string(),
                message: format!(
                    "Rank mismatch: expected {}, got {}",
                    self.rank, registration.rank
                ),
                scheduler_protocol_version: Some("0.1.0".to_string()),
                assigned_worker_id: None,
            });
        }

        // Accept registration
        WorkerResponse::RegisterAck(WorkerRegistrationAck {
            status: "ok".to_string(),
            message: format!("Worker {} registered successfully", self.worker.worker_id()),
            scheduler_protocol_version: Some("0.1.0".to_string()),
            assigned_worker_id: None,
        })
    }

    /// Handle LoadModel command
    fn handle_load_model(&mut self, params: ModelLoadParams) -> WorkerResponse {
        println!("[Worker-{}] Loading model from {}", self.rank, params.model_path);
        let start_time = Instant::now();

        // Determine device type
        #[cfg(feature = "cuda")]
        let device_type = DeviceType::Cuda(params.device_id as i32);
        #[cfg(not(feature = "cuda"))]
        let device_type = DeviceType::Cpu;

        // Create model using factory (for now, only Llama3 is supported)
        let model = match ModelFactory::create_llama3(
            std::path::Path::new(&params.model_path),
            device_type,
            false, // is_quant - TODO: detect from dtype
        ) {
            Ok(m) => Box::new(m) as Box<dyn crate::model::Model>,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ModelLoadFailed,
                    format!("Failed to create model: {}", e),
                );
            }
        };

        // Load model into worker
        if let Err(e) = self.worker.load_model(model) {
            return self.make_error(
                ErrorCode::ModelLoadFailed,
                format!("Failed to load model into worker: {}", e),
            );
        }

        let load_time_ms = start_time.elapsed().as_millis() as u64;

        // Get memory stats
        let _ = self.worker.refresh_memory_stats();
        let memory_stats = self.worker.memory_stats();

        // Get model parameters count
        let num_parameters = self.worker.model()
            .map(|m| m.config().estimate_num_parameters())
            .unwrap_or(0);

        WorkerResponse::ModelLoaded(ModelLoadedInfo {
            worker_id: self.worker.worker_id().to_string(),
            device_id: params.device_id,
            model_name: "llama3".to_string(), // TODO: Get from model config
            num_parameters,
            memory_used: memory_stats.used, // Total used memory after model loading
            tp_rank: params.tp_rank,
            tp_world_size: params.tp_world_size,
            load_time_ms,
        })
    }

    /// Handle Profile command
    fn handle_profile(&mut self, params: ProfileParams) -> WorkerResponse {
        println!("[Worker-{}] Profiling memory...", self.rank);

        // Ensure model is loaded
        if !self.worker.is_model_loaded() {
            return self.make_error(
                ErrorCode::InvalidState,
                "Model not loaded, cannot profile".to_string(),
            );
        }

        // Refresh memory stats
        if let Err(e) = self.worker.refresh_memory_stats() {
            return self.make_error(
                ErrorCode::DeviceError,
                format!("Failed to get memory stats: {}", e),
            );
        }

        let memory_stats = self.worker.memory_stats();

        // Calculate available memory for KV cache
        // Leave some headroom for activations (e.g., 20%)
        let headroom = (memory_stats.total as f64 * 0.2) as u64;
        let available_kv_cache = memory_stats.free.saturating_sub(headroom);

        WorkerResponse::ProfileCompleted(ProfileResult {
            peak_memory_forward: memory_stats.workspace_memory,
            memory_model: memory_stats.model_memory,
            total_memory: memory_stats.total,
            available_kv_cache_memory: available_kv_cache,
            avg_prefill_time_ms: 0.0, // TODO: Actually run profiling
            avg_decode_time_ms: 0.0,
            profiled_batch_size: params.batch_size,
            profiled_seq_len: params.seq_len,
        })
    }

    /// Handle InitKVCache command
    fn handle_init_kv_cache(&mut self, params: InitKVCacheParams) -> WorkerResponse {
        println!(
            "[Worker-{}] Initializing KV Cache: {} blocks x {} size",
            self.rank, params.num_blocks, params.block_size
        );
        let start_time = Instant::now();

        // Ensure model is loaded
        if !self.worker.is_model_loaded() {
            return self.make_error(
                ErrorCode::InvalidState,
                "Model not loaded, cannot init KV cache".to_string(),
            );
        }

        // Initialize KV cache
        if let Err(e) = self.worker.init_kv_cache(params.num_blocks, params.block_size) {
            return self.make_error(
                ErrorCode::KVCacheInitFailed,
                format!("Failed to init KV cache: {}", e),
            );
        }

        let init_time_ms = start_time.elapsed().as_millis() as u64;

        // Get KV cache info
        let kv_cache = self.worker.kv_cache().expect("KV cache just initialized");
        let memory_used = kv_cache.memory_bytes() as u64;
        let bytes_per_block = (params.block_size
            * (params.num_heads as usize)
            * (params.head_dim as usize)
            * 2 // K + V
            * 2) as u64; // BF16 = 2 bytes

        WorkerResponse::KVCacheInitialized(KVCacheInfo {
            allocated_blocks: params.num_blocks,
            memory_used,
            bytes_per_block,
            total_capacity_tokens: params.num_blocks * params.block_size,
            init_time_ms,
        })
    }

    /// Handle Forward command
    fn handle_forward(&mut self, params: ForwardParams) -> WorkerResponse {
        let start_time = Instant::now();
        let batch_size = params.token_ids.len();

        // Ensure model and KV cache are ready
        if !self.worker.is_model_loaded() {
            return self.make_error(
                ErrorCode::InvalidState,
                "Model not loaded".to_string(),
            );
        }
        if !self.worker.is_kv_cache_initialized() {
            return self.make_error(
                ErrorCode::InvalidState,
                "KV cache not initialized".to_string(),
            );
        }

        // For now, handle single sequence (batch_size = 1)
        // TODO: Support batched inference
        if batch_size != 1 {
            return self.make_error(
                ErrorCode::InvalidParams,
                format!("Batch size {} not supported yet, use 1", batch_size),
            );
        }

        let token_ids = &params.token_ids[0];
        let position_ids = &params.position_ids[0];
        let device_type = self.worker.device_type();

        // Create input tensor from token_ids
        let input_tokens = match create_i32_tensor(token_ids, device_type) {
            Ok(t) => t,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Failed to create input tensor: {}", e),
                );
            }
        };

        // Create position tensor from position_ids
        let positions = match create_i32_tensor(position_ids, device_type) {
            Ok(t) => t,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Failed to create position tensor: {}", e),
                );
            }
        };

        // Execute forward pass
        let logits = match self.worker.forward(&input_tokens, &positions) {
            Ok(l) => l,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Forward pass failed: {}", e),
                );
            }
        };

        // Sample next token
        let mut output_token = match Tensor::new(&[1], crate::base::DataType::I32, device_type) {
            Ok(t) => t,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Failed to create output tensor: {}", e),
                );
            }
        };

        if let Err(e) = self.worker.sample(&logits, &mut output_token) {
            return self.make_error(
                ErrorCode::ForwardFailed,
                format!("Sampling failed: {}", e),
            );
        }

        // Get sampled token ID
        // First move tensor to CPU if needed, then extract data
        let cpu_output = match output_token.to_cpu() {
            Ok(t) => t,
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Failed to move output to CPU: {}", e),
                );
            }
        };
        let next_token_id = match cpu_output.as_i32().and_then(|t| t.as_slice()) {
            Ok(v) => v[0],
            Err(e) => {
                return self.make_error(
                    ErrorCode::ForwardFailed,
                    format!("Failed to get output token: {}", e),
                );
            }
        };

        let inference_time_ms = start_time.elapsed().as_millis() as u64;

        WorkerResponse::ForwardCompleted(ForwardResult {
            next_token_ids: vec![next_token_id],
            finished: vec![false], // TODO: Check for EOS
            logits: None, // TODO: Support return_logits
            inference_time_ms,
            num_kv_blocks_used: params.kv_cache_block_ids[0].len(),
        })
    }

    /// Handle GetStatus command
    fn handle_get_status(&self) -> WorkerResponse {
        let memory_stats = self.worker.memory_stats();
        let perf_stats = self.worker.perf_stats();
        let config = self.worker.config();

        let state = match self.worker.state() {
            WorkerState::Uninitialized => ProtocolWorkerState::Initializing,
            WorkerState::Ready => ProtocolWorkerState::Idle,
            WorkerState::Processing => ProtocolWorkerState::Inferencing,
            WorkerState::Error => ProtocolWorkerState::Error,
        };

        WorkerResponse::Status(WorkerStatus {
            worker_id: self.worker.worker_id().to_string(),
            device_id: config.device_id,
            state,
            model_loaded: self.worker.is_model_loaded(),
            kv_cache_initialized: self.worker.is_kv_cache_initialized(),
            memory_stats: ProtocolMemoryStats {
                total: memory_stats.total,
                used: memory_stats.used,
                free: memory_stats.free,
                model_memory: memory_stats.model_memory,
                kv_cache_memory: memory_stats.kv_cache_memory,
                activation_memory: memory_stats.workspace_memory,
            },
            performance_stats: ProtocolPerformanceStats {
                total_requests: perf_stats.total_requests,
                total_tokens: perf_stats.total_tokens,
                avg_prefill_time_ms: perf_stats.avg_prefill_ms(),
                avg_decode_time_ms: perf_stats.avg_decode_per_token_ms(),
                throughput_tokens_per_sec: perf_stats.throughput_tps(),
                gpu_utilization: 0.0, // TODO: Track GPU utilization
            },
            tp_rank: Some(config.tp_rank),
            tp_world_size: Some(config.tp_world_size),
        })
    }

    /// Handle UnloadModel command
    fn handle_unload_model(&mut self) -> WorkerResponse {
        println!("[Worker-{}] Unloading model...", self.rank);

        // Reset KV cache if initialized
        if self.worker.is_kv_cache_initialized() {
            let _ = self.worker.reset_kv_cache();
        }

        // Note: Actual model unloading would require Worker to support it
        // For now, we just acknowledge the command

        WorkerResponse::ModelUnloaded
    }

    /// Send response back to Scheduler
    async fn send_response(&mut self, response: WorkerResponse) -> Result<()> {
        let payload = bincode::serialize(&response)?;

        // Safety check: ensure payload is not empty
        if payload.is_empty() {
            eprintln!("[Worker-{}] WARNING: Attempting to send empty payload!", self.rank);
            return Err(anyhow::anyhow!("Cannot send empty response payload"));
        }

        // DealerSocket 发送格式：[empty_frame, payload]
        // RouterSocket 接收格式：[address, empty_frame, payload]
        let mut msg = ZmqMessage::try_from(Vec::<u8>::new())?; // empty delimiter frame
        msg.push_back(payload.into());
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Create an error response
    fn make_error(&self, code: ErrorCode, message: String) -> WorkerResponse {
        eprintln!("[Worker-{}] Error: {:?} - {}", self.rank, code, message);
        WorkerResponse::Error(WorkerError {
            code,
            message,
            details: None,
            worker_id: self.worker.worker_id().to_string(),
            device_id: self.worker.config().device_id,
        })
    }

    /// Get worker rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        self.running
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_server_config() {
        // Basic configuration test
        let rank = 0;
        let world_size = 2;
        assert!(rank < world_size);
    }
}
