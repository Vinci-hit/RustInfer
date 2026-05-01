use clap::Parser;
use std::sync::Arc;
use std::thread;

use infer_worker::base::DeviceType;
use infer_worker::worker::{ModelRunner, SharedBuffers, WorkerServer};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker")]
#[command(about = "RustInfer Worker - 单 GPU 推理进程")]
struct Args {
    /// 模型路径
    #[arg(short, long)]
    model: String,

    /// 模型类型: llama3 或 qwen3
    #[arg(long, default_value = "llama3")]
    model_type: String,

    /// GPU 设备: cuda:0, cuda:1, ...
    #[arg(short, long, default_value = "cuda:0")]
    device: String,

    /// ZMQ 输入端口 (调度器 PUSH → Worker PULL)
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-0-in.ipc")]
    zmq_in: String,

    /// ZMQ 输出端口 (Worker PUSH → 调度器 PULL)
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-0-out.ipc")]
    zmq_out: String,

    /// 单步最大 token 数
    #[arg(long, default_value = "2048")]
    max_batch_tokens: usize,

    /// 最大并发序列数
    #[arg(long, default_value = "64")]
    max_num_seqs: usize,

    /// 日志级别
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .init();

    tracing::info!("RustInfer Worker starting...");
    tracing::info!("  Model: {} ({})", args.model, args.model_type);
    tracing::info!("  Device: {}", args.device);
    tracing::info!("  ZMQ IN:  {}", args.zmq_in);
    tracing::info!("  ZMQ OUT: {}", args.zmq_out);
    tracing::info!("  Max batch tokens: {}", args.max_batch_tokens);
    tracing::info!("  Max num seqs: {}", args.max_num_seqs);

    // 解析 device
    let device_id = parse_device_id(&args.device).expect("Invalid device string");
    let device = DeviceType::Cuda(device_id);

    // TODO: 加载模型
    // let model = load_model(&args.model, &args.model_type, device);
    tracing::info!("Loading model...");

    // 加载模型
    let model = infer_worker::model::llm::llama3::Llama3::new(&args.model, device)
        .expect("Failed to load model");
    tracing::info!("Model loaded");

    // 创建 InferenceState 数组 (每个 slot 一个)
    let mut states = Vec::with_capacity(args.max_num_seqs);
    for _ in 0..args.max_num_seqs {
        states.push(model.create_state().expect("Failed to create InferenceState"));
    }
    tracing::info!("Created {} inference states", states.len());

    // 预分配共享 buffer
    let shared = SharedBuffers::new(args.max_batch_tokens, args.max_num_seqs, device)
        .expect("Failed to allocate shared buffers");
    tracing::info!("Shared buffers allocated");

    // ZMQ sockets
    let zmq_ctx = zmq::Context::new();

    let zmq_in = zmq_ctx.socket(zmq::PULL).expect("Failed to create ZMQ PULL socket");
    zmq_in.bind(&args.zmq_in).expect("Failed to bind ZMQ PULL");

    let zmq_out = zmq_ctx.socket(zmq::PUSH).expect("Failed to create ZMQ PUSH socket");
    zmq_out.connect(&args.zmq_out).expect("Failed to connect ZMQ PUSH");

    tracing::info!("ZMQ sockets ready");

    // 创建 Runner 和 Server
    let runner_shared = Arc::clone(&shared);
    let runner = ModelRunner::new(model, states, runner_shared, device_id)
        .expect("Failed to create ModelRunner");

    let server = WorkerServer::new(zmq_in, zmq_out, shared, device_id, 128009); // Llama3 EOS

    // 启动 Runner 线程
    let runner_handle = thread::Builder::new()
        .name("model-runner".into())
        .spawn(move || runner.run())
        .expect("Failed to spawn runner thread");

    tracing::info!("Worker pipeline running.");

    // Server 在主线程运行
    server.run();

    // 正常不会到这里
    runner_handle.join().unwrap();
}

fn parse_device_id(s: &str) -> Result<i32, String> {
    if let Some(id_str) = s.strip_prefix("cuda:") {
        id_str
            .parse::<i32>()
            .map_err(|e| format!("Invalid device id: {}", e))
    } else {
        Err(format!("Invalid device: {}. Use 'cuda:0', 'cuda:1', etc.", s))
    }
}
