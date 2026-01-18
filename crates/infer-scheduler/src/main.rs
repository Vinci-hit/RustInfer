//! RustInfer Scheduler - 主服务入口
//!
//! 负责启动调度器服务，协调 GPU Workers 执行推理任务

use infer_scheduler::config::SchedulerConfig;
use infer_scheduler::coordinator::Coordinator;
use infer_scheduler::policy::ContinuousBatchingPolicy;
use infer_scheduler::transport::{create_frontend_channel, WorkerProxy};
use infer_protocol::{ModelLoadParams, ProfileParams, InitKVCacheParams};

use anyhow::{Result, Context};
use tracing::{info, warn};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // 加载配置（支持命令行参数和配置文件）
    let config = SchedulerConfig::load()
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
    let (_frontend_tx, frontend_rx) = create_frontend_channel();

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
            // TODO: 将 tokenizer_path 从 Scheduler 配置中移除
            // 目前保留是为了兼容 Worker 的接口要求
            // 未来 Worker 应该自己从 model_path 加载 tokenizer.json
            tokenizer_path: None,
            enable_flash_attn: config.model.enable_flash_attn,
            custom_config: config.model.custom_config.clone(),
        };

        let model_info = worker_proxy
            .load_model(&worker_id, model_params)
            .await
            .context("Failed to load model")?;

        info!(
            "✅ Model loaded: {} MB, parameters={}",
            model_info.memory_used / 1024 / 1024,
            model_info.num_parameters
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
            "✅ Profile completed: {} MB total, {} MB available for KV Cache",
            profile_result.total_memory / 1024 / 1024,
            profile_result.available_kv_cache_memory / 1024 / 1024
        );

        // Step 3: 初始化 KV Cache
        info!("🗄️  Initializing KV Cache...");

        // 使用从 config.json 读取的真实参数
        let kv_cache_params = InitKVCacheParams {
            num_blocks: config.memory.total_blocks,
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
    );

    // 设置默认 Worker ID
    if !registered_workers.is_empty() {
        coordinator.set_default_worker(registered_workers[0].worker_id.clone());
    }

    info!("✅ Coordinator created successfully");
    info!("📊 Statistics:");
    info!("  - Block size: {}", config.memory.block_size);
    info!("  - Total blocks: {}", config.memory.total_blocks);
    info!("  - Memory: {} MB", config.memory.total_blocks * config.memory.block_size * 2 / 1024); // 假设 bf16

    // 启动 Coordinator 主循环
    info!("🚀 Starting Coordinator main loop...");
    info!("💡 Scheduler is ready to accept requests");

    coordinator.run().await;

    Ok(())
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
