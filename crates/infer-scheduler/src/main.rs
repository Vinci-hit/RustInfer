use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_scheduler::Scheduler;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-scheduler")]
#[command(about = "RustInfer Scheduler — continuous batching 调度器")]
struct Args {
    /// ZMQ 前端地址（对接 HTTP Server, ROUTER socket）
    #[arg(long, default_value = "ipc:///tmp/rustinfer.ipc")]
    frontend_endpoint: String,

    /// ZMQ Worker PUSH 地址（发 PrefillBatchCmd）
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-in.ipc")]
    worker_push_endpoint: String,

    /// ZMQ Worker PULL 地址（收 StepOutput）
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-out.ipc")]
    worker_pull_endpoint: String,

    /// 最大 batch tokens
    #[arg(long, default_value = "1024")]
    max_batch_tokens: usize,

    /// 最大 batch seqs (= slot 数量)
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// 日志级别
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化日志
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("RustInfer Scheduler starting...");
    tracing::info!("  Frontend: {}", args.frontend_endpoint);
    tracing::info!("  Worker PUSH: {}", args.worker_push_endpoint);
    tracing::info!("  Worker PULL: {}", args.worker_pull_endpoint);
    tracing::info!("  max_batch_seqs: {}", args.max_batch_seqs);
    tracing::info!("  max_batch_tokens: {}", args.max_batch_tokens);

    // 创建 ZMQ sockets
    let zmq_ctx = zmq::Context::new();

    // Frontend ROUTER（对接 HTTP Server 的 DEALER）
    let zmq_frontend = zmq_ctx.socket(zmq::ROUTER)?;
    zmq_frontend.bind(&args.frontend_endpoint)?;
    tracing::info!("Frontend ROUTER bound to {}", args.frontend_endpoint);

    // Worker PUSH（发 PrefillBatchCmd）
    let zmq_to_worker = zmq_ctx.socket(zmq::PUSH)?;
    zmq_to_worker.bind(&args.worker_push_endpoint)?;
    tracing::info!("Worker PUSH bound to {}", args.worker_push_endpoint);

    // Worker PULL（收 StepOutput）
    let zmq_from_worker = zmq_ctx.socket(zmq::PULL)?;
    zmq_from_worker.bind(&args.worker_pull_endpoint)?;
    tracing::info!("Worker PULL bound to {}", args.worker_pull_endpoint);

    // 启动 Scheduler
    let scheduler = Scheduler::new(
        zmq_frontend,
        zmq_to_worker,
        zmq_from_worker,
        args.max_batch_tokens,
        args.max_batch_seqs,
    );

    tracing::info!("Scheduler running...");
    scheduler.run();

    Ok(())
}
