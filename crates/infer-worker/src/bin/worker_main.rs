//! RustInfer Worker 进程入口。
//!
//! 启动 ModelRunner 线程 + WorkerServer 线程，通过 ZMQ 与 Scheduler 通信。

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_worker::base::DeviceType;
use infer_worker::model::llm::LlmModel;
use infer_worker::worker::runner::ModelRunner;
use infer_worker::worker::WorkerServer;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker")]
#[command(about = "RustInfer Worker — GPU 推理进程")]
struct Args {
    /// 模型路径
    #[arg(short, long)]
    model: String,

    /// 模型类型: llama3 或 qwen3
    #[arg(long, default_value = "llama3")]
    model_type: String,

    /// 设备: cpu 或 cuda:0
    #[arg(short, long, default_value = "cuda:0")]
    device: String,

    /// ZMQ PULL 地址（收 Scheduler 的 PrefillBatchCmd）
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-in.ipc")]
    worker_pull_endpoint: String,

    /// ZMQ PUSH 地址（发 StepOutput 给 Scheduler）
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-out.ipc")]
    worker_push_endpoint: String,

    /// 最大 batch tokens
    #[arg(long, default_value = "1024")]
    max_batch_tokens: usize,

    /// 最大 batch seqs
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// 日志级别
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("RustInfer Worker starting...");
    tracing::info!("  Model: {}", args.model);
    tracing::info!("  Model type: {}", args.model_type);
    tracing::info!("  Device: {}", args.device);

    let device = parse_device(&args.device)?;

    // ZMQ sockets
    let zmq_ctx = zmq::Context::new();
    let zmq_pull = zmq_ctx.socket(zmq::PULL)?;
    zmq_pull.connect(&args.worker_pull_endpoint)?;
    tracing::info!("Worker PULL connected to {}", args.worker_pull_endpoint);

    let zmq_push = zmq_ctx.socket(zmq::PUSH)?;
    zmq_push.connect(&args.worker_push_endpoint)?;
    tracing::info!("Worker PUSH connected to {}", args.worker_push_endpoint);

    // 加载模型并启动
    match args.model_type.to_lowercase().as_str() {
        "llama3" | "llama" => {
            let model = infer_worker::model::llm::llama3::Llama3::new(&args.model, device)?;
            run_worker(model, device, zmq_pull, zmq_push, args.max_batch_tokens, args.max_batch_seqs)
        }
        "qwen3" | "qwen" => {
            let model = infer_worker::model::llm::qwen3::Qwen3::new(&args.model, device)?;
            run_worker(model, device, zmq_pull, zmq_push, args.max_batch_tokens, args.max_batch_seqs)
        }
        _ => anyhow::bail!("Unsupported model type: {}. Use 'llama3' or 'qwen3'.", args.model_type),
    }
}

fn run_worker<M: LlmModel + 'static>(
    model: M,
    device: DeviceType,
    zmq_pull: zmq::Socket,
    zmq_push: zmq::Socket,
    max_batch_tokens: usize,
    max_batch_seqs: usize,
) -> Result<()> {
    let eos_token_ids: Vec<i32> = model.tokenizer().eos_token_ids()
        .iter().map(|&id| id as i32).collect();

    tracing::info!("Model loaded, creating runner (max_batch_tokens={}, max_batch_seqs={})",
        max_batch_tokens, max_batch_seqs);

    let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);

    // Runner 线程
    let runner_loop = Arc::clone(&runner);
    let runner_handle = std::thread::spawn(move || runner_loop.run());

    // Server (当前线程)
    tracing::info!("Worker running...");
    let server = WorkerServer::new(
        Arc::clone(&runner),
        device,
        zmq_pull,
        zmq_push,
        eos_token_ids,
    );
    server.run();

    runner.request_shutdown();
    let _ = runner_handle.join();
    Ok(())
}

fn parse_device(s: &str) -> Result<DeviceType> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(DeviceType::Cpu),
        s if s.starts_with("cuda:") => {
            let id: i32 = s[5..].parse()?;
            Ok(DeviceType::Cuda(id))
        }
        _ => anyhow::bail!("Invalid device: {}. Use 'cpu' or 'cuda:0'", s),
    }
}
