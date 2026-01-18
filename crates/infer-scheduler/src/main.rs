//! RustInfer Scheduler - 主服务入口
//!
//! 负责启动调度器服务，协调 GPU Workers 执行推理任务

use infer_scheduler::config::SchedulerConfig;
use infer_scheduler::coordinator::Coordinator;
use infer_scheduler::policy::ContinuousBatchingPolicy;
use infer_scheduler::transport::{create_frontend_channel, WorkerProxy, ZmqFrontendServer};
use infer_protocol::{ModelLoadParams, ProfileParams, InitKVCacheParams, SchedulerOutput};

use anyhow::{Result, Context};
use tracing::{info, warn, error};
use tracing_subscriber;
use tokio::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // 加载配置（支持命令行参数和配置文件）
    let mut config = SchedulerConfig::load()
        .context("Failed to load configuration")?;

    // 初始化日志
    init_logging(&config.logging)?;

    info!("🚀 RustInfer Scheduler starting...");

    // 打印配置摘要
    config.print_summary();

    // 读取模型元数据（从 config.json）
    info!("📖 Reading model metadata from config.json...");
    let model_metadata = config.read_model_metadata()
        .context("Failed to read model metadata")?;

    info!(
        "✅ Model metadata: {} layers, {} heads, eos_token={}",
        model_metadata.num_layers,
        model_metadata.num_attention_heads,
        model_metadata.eos_token_id
    );

    // 创建前端通道（用于接收推理请求）
    let (frontend_tx, frontend_rx) = create_frontend_channel();

    // 创建共享的输出路由表 (request_id -> output_channel)
    // ZmqFrontendServer 和 Coordinator 将共享这个 map
    let output_router: Arc<RwLock<HashMap<String, mpsc::UnboundedSender<SchedulerOutput>>>> =
        Arc::new(RwLock::new(HashMap::new()));

    // 创建输出通道（用于ZMQ Frontend发送响应到Server）
    let (output_tx, _output_rx) = mpsc::unbounded_channel::<SchedulerOutput>();

    // 启动 ZMQ Frontend Server
    info!("📡 Binding ZMQ Frontend to: {}", config.network.frontend_endpoint);
    let zmq_frontend = ZmqFrontendServer::bind(
        config.network.frontend_endpoint.clone(),
        output_tx,
        frontend_tx,
        output_router.clone(),
    )
    .await
    .context("Failed to bind ZMQ Frontend Server")?;

    info!("✅ ZMQ Frontend Server bound successfully");

    // 启动 ZMQ 接收循环（在后台运行）
    zmq_frontend.start_loop();

    // 创建 Worker 代理
    info!("📡 Binding to Worker endpoint: {}", config.network.worker_endpoint);
    let mut worker_proxy = WorkerProxy::new(
        config.network.worker_endpoint.clone(),
        config.network.worker_timeout_ms,
    )
    .await
    .context("Failed to create WorkerProxy")?;

    info!("✅ Worker endpoint bound successfully: {}", worker_proxy.endpoint());

    // 等待 Worker 注册
    info!("⏳ Waiting for {} Worker(s) to register...", config.network.num_workers);
    let mut registered_workers = Vec::new();

    for i in 1..=config.network.num_workers {
        let worker_info = worker_proxy
            .wait_for_registration()
            .await
            .context(format!("Failed to register Worker {}/{}", i, config.network.num_workers))?;

        info!(
            "✅ Worker {}/{} registered: {} (rank={}, device={}:{})",
            i,
            config.network.num_workers,
            worker_info.worker_id,
            worker_info.rank,
            worker_info.device_type,
            worker_info.device_id
        );

        registered_workers.push(worker_info);
    }

    // 对于单 Worker 模式，立即初始化
    if config.network.num_workers == 1 {
        let worker_id = registered_workers[0].worker_id.clone();
        info!("🔧 Initializing single Worker: {}", worker_id);

        // Step 1: 加载模型
        info!("📦 Loading model...");
        let model_params = ModelLoadParams {
            device_id: registered_workers[0].device_id,
            model_path: config.model.model_path.clone(),
            dtype: config.model.dtype.clone(),
            tp_rank: 0,
            tp_world_size: config.parallelism.tp_size as u32,
            pp_rank: 0,
            pp_world_size: config.parallelism.pp_size as u32,
            tokenizer_path: None,
            enable_flash_attn: config.model.enable_flash_attn,
            custom_config: config.model.custom_config.clone(),
        };

        let model_info = worker_proxy
            .load_model(&worker_id, model_params)
            .await;

        // Handle load errors robustly - don't crash if Worker fails
        let model_info = match model_info {
            Ok(info) => info,
            Err(e) => {
                error!("Failed to load model on Worker {}: {}", worker_id, e);
                // Worker will be marked as failed through the connection manager
                return Err(e);
            }
        };

        info!(
            "✅ Model loaded: {:.2} GB, parameters={:.2}B",
            model_info.memory_used as f64 / (1024.0 * 1024.0 * 1024.0),
            model_info.num_parameters as f64 / 1_000_000_000.0
        );

        // Step 2: Profile 显存
        info!("🔍 Profiling GPU memory...");
        let profile_params = ProfileParams {
            batch_size: config.scheduling.max_batch_size,
            seq_len: 2048, // 默认序列长度
            num_rounds: 3,
            include_prefill: true,
            include_decode: true,
        };

        let profile_result = worker_proxy
            .profile(&worker_id, profile_params)
            .await
            .context("Failed to profile GPU memory")?;

        info!(
            "✅ Profile completed: {:.2} GB total, {:.2} GB available for KV Cache",
            profile_result.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            profile_result.available_kv_cache_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        // Step 2.5: 如果 total_blocks=0，根据 profile 结果自动计算
        let total_blocks = if config.memory.total_blocks == 0 {
            // 计算每个 block 需要的显存
            // = block_size * num_layers * num_kv_heads * head_dim * 2 (K和V) * dtype_bytes
            let dtype_bytes = match config.model.dtype.as_str() {
                "bf16" | "fp16" => 2,
                "fp32" => 4,
                _ => 2, // 默认 bf16
            };

            let bytes_per_block = config.memory.block_size
                * model_metadata.num_layers
                * model_metadata.num_kv_heads
                * model_metadata.head_dim
                * 2  // K 和 V
                * dtype_bytes;

            let computed_blocks = profile_result.available_kv_cache_memory as usize / bytes_per_block;

            // 应用 gpu_memory_utilization 系数
            let final_blocks = (computed_blocks as f32 * config.memory.gpu_memory_utilization) as usize;

            info!(
                "📊 Auto-computed total_blocks: {} (from {:.2} GB available memory)",
                final_blocks,
                profile_result.available_kv_cache_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            final_blocks
        } else {
            info!("📊 Using user-specified total_blocks: {}", config.memory.total_blocks);
            config.memory.total_blocks
        };

        // 更新 config 中的 total_blocks
        config.memory.total_blocks = total_blocks;

        // Step 3: 初始化 KV Cache
        info!("🗄️  Initializing KV Cache...");

        // 使用从 config.json 读取的真实参数
        let kv_cache_params = InitKVCacheParams {
            num_blocks: total_blocks,
            block_size: config.memory.block_size,
            num_layers: model_metadata.num_layers as u32,
            num_heads: model_metadata.num_kv_heads as u32,
            head_dim: model_metadata.head_dim as u32,
            dtype: config.model.dtype.clone(),
            use_unified_memory_pool: true,
        };

        let kv_cache_info = worker_proxy
            .init_kv_cache(&worker_id, kv_cache_params)
            .await
            .context("Failed to initialize KV Cache")?;

        info!(
            "✅ KV Cache initialized: {} blocks, {} MB",
            kv_cache_info.allocated_blocks,
            kv_cache_info.memory_used / 1024 / 1024
        );

        // 健康检查
        if worker_proxy.health_check(&worker_id).await? {
            info!("✅ Worker health check passed");
        } else {
            warn!("⚠️  Worker health check failed");
        }
    } else {
        // 多 Worker 模式：TODO 支持 Tensor Parallel
        warn!("⚠️  Multi-worker mode not yet implemented, using first worker only");
    }

    // 创建 Coordinator
    info!("🎯 Creating Coordinator...");

    // 使用配置转换方法
    let policy_config = config.to_policy_config();
    let policy = Box::new(ContinuousBatchingPolicy::new(policy_config));

    let coordinator_config = config.to_coordinator_config();

    let mut coordinator = Coordinator::new(
        policy,
        worker_proxy,
        frontend_rx,
        coordinator_config,
        output_router.clone(),
    );

    // 设置默认 Worker ID
    if !registered_workers.is_empty() {
        coordinator.set_default_worker(registered_workers[0].worker_id.clone());
    }

    info!("✅ Coordinator created successfully");
    info!("📊 Statistics:");
    info!("  - Block size: {}", config.memory.block_size);
    info!("  - Total blocks: {}", config.memory.total_blocks);
    info!("  - Memory: {} G", config.memory.total_blocks * config.memory.block_size
                * model_metadata.num_layers
                * model_metadata.num_kv_heads
                * model_metadata.head_dim
                * 2  // K 和 V
                * 2 / 1024 / 1024 / 1024); // 假设 bf16

    // 启动 Coordinator 主循环
    info!("🚀 Starting Coordinator main loop...");
    info!("💡 Scheduler is ready to accept requests");

    // 使用 tokio::select! 来同时等待 coordinator 和 shutdown signal
    tokio::select! {
        _ = coordinator.run() => {
            info!("Coordinator exited normally");
        }
        _ = shutdown_signal() => {
            info!("Shutdown signal received, stopping Coordinator...");
        }
    }

    // 清理工作
    info!("Cleaning up resources...");
    drop(zmq_frontend);
    info!("✅ Scheduler shutdown complete");

    Ok(())
}

/// Graceful Shutdown Signal
async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => {
            info!("Received Ctrl+C signal");
        }
        Err(err) => {
            error!("Unable to listen for shutdown signal: {}", err);
        }
    }
}

/// 初始化日志系统
fn init_logging(logging_config: &infer_scheduler::config::LoggingConfig) -> Result<()> {
    let filter = match logging_config.log_level.as_str() {
        "trace" => "trace",
        "debug" => "debug",
        "info" => "info",
        "warn" => "warn",
        "error" => "error",
        _ => {
            eprintln!("Invalid log level: {}, using 'info'", logging_config.log_level);
            "info"
        }
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    Ok(())
}
