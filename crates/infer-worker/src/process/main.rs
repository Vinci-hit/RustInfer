//! rustinfer-worker binary — entry point for the inference worker process.

use clap::Parser;

/// RustInfer Worker — GPU inference runtime with continuous batching.
#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker", version = "0.2.0")]
struct Args {
    /// Path to model weights (safetensors directory).
    #[arg(long)]
    model_path: String,

    /// CUDA device (e.g. "cuda:0").
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Model type: "llama3", "qwen3", or "zimage".
    #[arg(long, default_value = "qwen3")]
    model_type: String,

    /// Scheduler control plane endpoint.
    #[arg(long, default_value = "tcp://127.0.0.1:5500")]
    control_endpoint: String,

    /// Scheduler → Worker data plane endpoint (PULL).
    #[arg(long, default_value = "tcp://127.0.0.1:5501")]
    data_recv_endpoint: String,

    /// Worker → Scheduler data plane endpoint (PUSH).
    #[arg(long, default_value = "tcp://127.0.0.1:5502")]
    data_send_endpoint: String,

    /// Worker ID (unique per process).
    #[arg(long, default_value = "worker-0")]
    worker_id: String,

    /// Maximum batch tokens per step.
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: usize,

    /// Maximum sequences per batch.
    #[arg(long, default_value_t = 256)]
    max_batch_seqs: usize,

    /// Maximum sequence length (KV cache capacity).
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Heartbeat interval in milliseconds.
    #[arg(long, default_value_t = 5000)]
    heartbeat_interval_ms: u64,
}

fn main() {
    let args = Args::parse();

    eprintln!("rustinfer-worker v0.2.0");
    eprintln!("  worker_id: {}", args.worker_id);
    eprintln!("  model: {} ({})", args.model_path, args.model_type);
    eprintln!("  device: {}", args.device);
    eprintln!("  control: {}", args.control_endpoint);
    eprintln!("  data_recv: {}", args.data_recv_endpoint);
    eprintln!("  data_send: {}", args.data_send_endpoint);
    eprintln!("  batch: max_tokens={}, max_seqs={}", args.max_batch_tokens, args.max_batch_seqs);
    eprintln!();

    // 1. Initialize ZMQ context
    let zmq_ctx = zmq::Context::new();

    // 2. Connect control + data sockets
    let control = infer_worker::process::control_pump::ControlPump::new(
        &zmq_ctx, args.worker_id.clone(), &args.control_endpoint,
    ).expect("failed to connect control plane");

    let data = infer_worker::process::data_pump::DataPump::new(
        &zmq_ctx, &args.data_recv_endpoint, &args.data_send_endpoint,
    ).expect("failed to connect data plane");

    // 3. Send Hello
    control.send_hello().expect("failed to send hello");
    eprintln!("[bootstrap] Hello sent, waiting for LoadModel...");

    // 4. Wait for LoadModel command
    let (_load_msg, _req_id) = control.recv().expect("failed to recv LoadModel");
    eprintln!("[bootstrap] Received LoadModel, loading weights...");

    // 5. Load model (placeholder — actual loading depends on model_type + device)
    // In production:
    //   let device = Cuda::new(device_id)?;
    //   let model = WeightLoader::load_qwen3(&cfg, &device)?;
    //   let runner = ModelRunner::new(model, device, max_batch_seqs, max_seq_len)?;
    eprintln!("[bootstrap] Model loaded (stub).");

    // 6. Send Ready
    use infer_protocol::worker_to_scheduler_control::WorkerCapacity;
    control.send_ready(
        "default".into(),
        args.model_path.clone(),
        args.model_type.clone(),
        WorkerCapacity {
            max_batch_tokens: args.max_batch_tokens,
            max_batch_seqs: args.max_batch_seqs,
            max_running_requests: args.max_batch_seqs,
            max_total_kv_tokens: Some(args.max_seq_len * args.max_batch_seqs),
            free_mem_before_load_gb: None,
            free_mem_after_load_gb: None,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb: None,
            graph_mem_usage_gb: None,
        },
    ).expect("failed to send Ready");
    eprintln!("[bootstrap] Ready sent. Entering serve loop...");

    // 7. Enter serve loop
    // In production: run_serve_loop(runner_sync, control, data, config)
    // For now: simple recv-loop demonstrating the protocol
    let config = infer_worker::process::serve_loop::ServeConfig {
        max_batch_tokens: args.max_batch_tokens,
        max_batch_seqs: args.max_batch_seqs,
        heartbeat_interval_ms: args.heartbeat_interval_ms,
    };
    eprintln!("[serve] Config: {:?}", (config.max_batch_tokens, config.max_batch_seqs));
    eprintln!("[serve] Waiting for batch commands...");

    // Placeholder loop — full integration requires a loaded model
    loop {
        match data.try_recv_batch(config.heartbeat_interval_ms as i64) {
            Ok(Some(cmd)) => {
                eprintln!("[serve] Received batch command: {:?}", std::mem::discriminant(&cmd));
            }
            Ok(None) => {
                let _ = control.send_heartbeat(0);
            }
            Err(e) => {
                eprintln!("[serve] Error: {}", e);
                break;
            }
        }
    }
}
