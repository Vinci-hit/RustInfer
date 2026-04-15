use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod engine;
mod scheduler;
mod zmq_server;

use engine::{InferenceEngine, ModelInstance};
use infer_core::base::DeviceType;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-engine")]
#[command(about = "RustInfer Engine - 专用推理进程", long_about = None)]
struct Args {
    /// 模型路径
    #[arg(short, long)]
    model: String,

    /// 模型类型: llama3 或 qwen3
    #[arg(long, default_value = "llama3")]
    model_type: String,

    /// 设备: cpu 或 cuda:0, cuda:1 等
    #[arg(short, long, default_value = "cuda:0")]
    device: String,

    /// ZMQ绑定地址
    #[arg(short, long, default_value = "ipc:///tmp/rustinfer.ipc")]
    zmq_endpoint: String,

    /// 最大batch size
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// 请求队列最大长度
    #[arg(long, default_value = "128")]
    max_queue_size: usize,

    /// 调度间隔 (毫秒)
    #[arg(long, default_value = "10")]
    schedule_interval_ms: u64,

    /// 日志级别
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化日志
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("🚀 RustInfer Engine starting...");
    tracing::info!("  Model: {}", args.model);
    tracing::info!("  Model Type: {}", args.model_type);
    tracing::info!("  Device: {}", args.device);
    tracing::info!("  ZMQ Endpoint: {}", args.zmq_endpoint);
    tracing::info!("  Max Batch Size: {}", args.batch_size);

    // 解析设备
    let device = parse_device(&args.device)?;

    // 加载模型
    tracing::info!("Loading model...");
    let model = match args.model_type.to_lowercase().as_str() {
        "llama3" | "llama" => {
            let m = infer_core::model::llama3::Llama3::new(&args.model, device, false)?;
            let state = m.create_state()?;
            ModelInstance::Llama3(m, state)
        }
        "qwen3" | "qwen" => {
            let m = infer_core::model::qwen3::Qwen3::new(&args.model, device, false)?;
            ModelInstance::Qwen3(m)
        }
        _ => {
            anyhow::bail!(
                "Unsupported model type: {}. Use 'llama3' or 'qwen3'.",
                args.model_type
            );
        }
    };
    tracing::info!("✅ Model loaded successfully!");

    // 创建Engine
    let engine = InferenceEngine::new(
        model,
        args.batch_size,
        args.max_queue_size,
        args.schedule_interval_ms,
    );

    // 启动ZMQ服务器
    tracing::info!("Starting ZMQ server...");
    zmq_server::run(engine, &args.zmq_endpoint).await?;

    Ok(())
}

fn parse_device(s: &str) -> Result<DeviceType> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(DeviceType::Cpu),
        s if s.starts_with("cuda:") => {
            let device_id: i32 = s[5..].parse()?;
            Ok(DeviceType::Cuda(device_id))
        }
        _ => Err(anyhow::anyhow!("Invalid device: {}. Use 'cpu' or 'cuda:0'", s)),
    }
}
