//! RustInfer Worker 进程入口。
//!
//! 启动 ModelRunner 线程 + SubScheduler 线程，通过 ZMQ 与 Scheduler 通信。

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_worker::base::DeviceType;
use infer_worker::model::llm::LlmModel;
use infer_protocol::scheduler_to_worker_control::{InitPagedKv, LoadModel, SchedulerControlMessage};
use infer_protocol::worker_to_scheduler_control::{PagedKvReady, WorkerCapacity, WorkerMemoryProfile, WorkerState};
use infer_worker::worker::control_client::WorkerControlClient;
use infer_worker::worker::runner::ModelRunner;
use infer_worker::worker::{DiffusionWorkerServer, SubScheduler};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker")]
#[command(about = "RustInfer Worker — GPU 推理进程")]
struct Args {
    /// 模型路径。未提供时进入 agent 模式，等待 Scheduler 通过控制面下发 LoadModel。
    #[arg(short, long)]
    model: Option<String>,

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

    /// ZMQ 控制面地址（向 Scheduler 上报生命周期状态）
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-control.ipc")]
    worker_control_endpoint: String,

    /// Worker ID（默认按 pid + device 生成）
    #[arg(long)]
    worker_id: Option<String>,

    /// 最大 batch tokens
    #[arg(long, default_value = "1024")]
    max_batch_tokens: usize,

    /// 最大 batch seqs
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// 日志级别
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Start CUDA profiler on the first submitted step and stop after N completed steps (0=disabled).
    #[arg(long, default_value_t = 0)]
    profile_cuda_steps: usize,
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
    tracing::info!("  Model: {}", args.model.as_deref().unwrap_or("<scheduler-assigned>"));
    tracing::info!("  Model type: {}", args.model_type);
    tracing::info!("  Device: {}", args.device);

    let worker_id = args
        .worker_id
        .clone()
        .unwrap_or_else(|| default_worker_id(&args.device));
    let control = WorkerControlClient::connect(&args.worker_control_endpoint, worker_id)?;
    control.send_hello(
        std::process::id(),
        hostname(),
        args.device.clone(),
    )?;
    control.send_progress(WorkerState::Connecting, "worker control plane connected")?;

    let device = parse_device(&args.device)?;
    control.send_progress(WorkerState::Registered, "worker registered locally")?;

    // ZMQ sockets
    let zmq_ctx = zmq::Context::new();
    let zmq_pull = zmq_ctx.socket(zmq::PULL)?;
    zmq_pull.connect(&args.worker_pull_endpoint)?;
    tracing::info!("Worker PULL connected to {}", args.worker_pull_endpoint);

    let zmq_push = zmq_ctx.socket(zmq::PUSH)?;
    zmq_push.connect(&args.worker_push_endpoint)?;
    tracing::info!("Worker PUSH connected to {}", args.worker_push_endpoint);

    let load_model = match cli_load_model(&args) {
        Some(cmd) => cmd,
        None => wait_for_load_model(&control)?,
    };

    if load_model.device != args.device {
        tracing::warn!(
            "LoadModel device {} differs from worker device {}; using worker device",
            load_model.device,
            args.device,
        );
    }

    // 加载模型并启动
    control.send_progress(WorkerState::LoadingModel, "loading model weights")?;
    match load_model.model_type.to_lowercase().as_str() {
        "llama3" | "llama" => {
            let model = infer_worker::model::llm::llama3::Llama3::new(&load_model.model_path, device)?;
            run_worker(
                model,
                device,
                args.device.clone(),
                load_model,
                zmq_pull,
                zmq_push,
                control,
                args.profile_cuda_steps,
            )
        }
        "qwen3" | "qwen" => {
            let model = infer_worker::model::llm::qwen3::Qwen3::new(&load_model.model_path, device)?;
            run_worker(
                model,
                device,
                args.device.clone(),
                load_model,
                zmq_pull,
                zmq_push,
                control,
                args.profile_cuda_steps,
            )
        }
        "zimage" | "z-image" | "z_image" | "z-image-turbo" => {
            let pipeline = infer_worker::model::diffusion::z_image::pipeline::ZImagePipeline::from_pretrained(
                &load_model.model_path,
                device,
            )?;
            run_diffusion_worker(
                pipeline,
                args.device.clone(),
                load_model,
                zmq_pull,
                zmq_push,
                control,
            )
        }
        _ => anyhow::bail!("Unsupported model type: {}. Use 'llama3', 'qwen3', or 'zimage'.", load_model.model_type),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker<M: LlmModel + 'static>(
    model: M,
    device: DeviceType,
    device_label: String,
    load_model: LoadModel,
    zmq_pull: zmq::Socket,
    zmq_push: zmq::Socket,
    control: WorkerControlClient,
    profile_cuda_steps: usize,
) -> Result<()> {
    let eos_token_ids: Vec<i32> = model.tokenizer().eos_token_ids()
        .iter().map(|&id| id as i32).collect();

    let max_batch_tokens = load_model.max_batch_tokens;
    let max_batch_seqs = load_model.max_batch_seqs;

    tracing::info!("Model loaded, creating runner (max_batch_tokens={}, max_batch_seqs={})",
        max_batch_tokens, max_batch_seqs);
    control.send_progress(WorkerState::AllocatingRuntime, "creating ModelRunner runtime")?;

    let runner = match ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs) {
        Ok(runner) => Arc::new(runner),
        Err(e) => {
            let _ = control.send_error(WorkerState::Error, format!("ModelRunner::new failed: {e}"));
            return Err(e);
        }
    };

    let mut max_total_kv_tokens = None;
    let mut free_mem_after_load_gb = None;
    let mut workspace_mem_usage_gb = None;

    if load_model.kv_cache_mode.as_deref().is_some_and(|mode| mode.starts_with("paged:")) {
        control.send_progress(WorkerState::ProfilingMemory, "profiling memory for paged KV")?;
        let profile = build_memory_profile(
            &runner,
            &control,
            &load_model,
            &device_label,
            device,
        )?;
        free_mem_after_load_gb = Some(bytes_to_gb(profile.free_mem_after_dummy_bytes));
        control.send_memory_profile(profile)?;

        let init = wait_for_init_paged_kv(&control, &load_model.model_instance_id)?;
        control.send_progress(
            WorkerState::AllocatingRuntime,
            format!("allocating paged KV pool: blocks={} block_size={}", init.initial_num_blocks, init.block_size),
        )?;
        runner.init_paged_kv_pool(init.block_size as usize, init.initial_num_blocks as usize)?;
        let (_, _, bytes_allocated) = runner
            .paged_kv_pool_summary()
            .ok_or_else(|| anyhow::anyhow!("paged KV pool summary missing after init"))?;
        workspace_mem_usage_gb = Some(bytes_to_gb(bytes_allocated as u64));
        max_total_kv_tokens = Some(init.initial_num_blocks as usize * init.block_size as usize);
        control.send_paged_kv_ready(PagedKvReady {
            worker_id: control.worker_id().to_string(),
            model_instance_id: load_model.model_instance_id.clone(),
            block_size: init.block_size,
            initial_num_blocks: init.initial_num_blocks,
            max_num_blocks: init.max_num_blocks,
            max_blocks_per_seq: init.max_blocks_per_seq,
            bytes_allocated: bytes_allocated as u64,
        })?;
    }

    control.send_progress(WorkerState::Warmup, "runtime allocated; ready for runner thread")?;
    control.send_ready(
        load_model.model_instance_id,
        load_model.model_path,
        load_model.model_type,
        device_label,
        WorkerCapacity {
            max_batch_tokens,
            max_batch_seqs,
            max_running_requests: max_batch_seqs,
            max_total_kv_tokens,
            free_mem_before_load_gb: None,
            free_mem_after_load_gb,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb,
            graph_mem_usage_gb: None,
        },
    )?;

    // Runner 线程
    let runner_loop = Arc::clone(&runner);
    let runner_handle = std::thread::spawn(move || runner_loop.run());

    // SubScheduler (当前线程)
    control.send_progress(WorkerState::Running, "worker data plane running")?;
    tracing::info!("Worker running...");
    let sub_scheduler = SubScheduler::new(
        Arc::clone(&runner),
        device,
        zmq_pull,
        zmq_push,
        eos_token_ids,
        profile_cuda_steps,
    );
    sub_scheduler.run();

    runner.request_shutdown();
    let _ = runner_handle.join();
    Ok(())
}

fn run_diffusion_worker<P: infer_worker::model::diffusion::pipeline::DiffusionPipeline + 'static>(
    pipeline: P,
    device_label: String,
    load_model: LoadModel,
    zmq_pull: zmq::Socket,
    zmq_push: zmq::Socket,
    control: WorkerControlClient,
) -> Result<()> {
    let max_batch_seqs = load_model.max_batch_seqs;
    control.send_progress(WorkerState::Warmup, "diffusion pipeline loaded; ready for data plane")?;
    control.send_ready(
        load_model.model_instance_id,
        load_model.model_path,
        load_model.model_type,
        device_label,
        WorkerCapacity {
            max_batch_tokens: load_model.max_batch_tokens,
            max_batch_seqs,
            max_running_requests: max_batch_seqs,
            max_total_kv_tokens: None,
            free_mem_before_load_gb: None,
            free_mem_after_load_gb: None,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb: None,
            graph_mem_usage_gb: None,
        },
    )?;

    control.send_progress(WorkerState::Running, "diffusion worker data plane running")?;
    tracing::info!("Diffusion worker running...");
    let server = DiffusionWorkerServer::new(pipeline, zmq_pull, zmq_push, max_batch_seqs);
    server.run();
    Ok(())
}

fn build_memory_profile<M: LlmModel>(
    runner: &Arc<ModelRunner<M>>,
    control: &WorkerControlClient,
    load_model: &LoadModel,
    device_label: &str,
    device: DeviceType,
) -> Result<WorkerMemoryProfile> {
    let (free_after_dummy_bytes, total_mem_bytes) = device_mem_info(device)?;
    let fraction = load_model
        .kv_cache_memory_fraction
        .unwrap_or(0.95)
        .clamp(0.0, 1.0);
    let suggested_kv_budget_bytes = (free_after_dummy_bytes as f64 * fraction as f64) as u64;
    let cfg = runner.model().config();
    let dtype_size = cfg.runtime_float_dtype(device)?.size_in_bytes() as u32;

    Ok(WorkerMemoryProfile {
        worker_id: control.worker_id().to_string(),
        model_instance_id: load_model.model_instance_id.clone(),
        device: device_label.to_string(),
        total_mem_bytes,
        free_mem_before_load_bytes: 0,
        free_mem_after_dummy_bytes: free_after_dummy_bytes,
        layer_num: cfg.layer_num as u32,
        kv_head_num: cfg.kv_head_num as u32,
        head_size: cfg.head_size as u32,
        dtype_size,
        max_batch_tokens: load_model.max_batch_tokens as u32,
        max_batch_seqs: load_model.max_batch_seqs as u32,
        max_model_len: load_model.max_model_len as u32,
        suggested_kv_budget_bytes,
    })
}

fn wait_for_init_paged_kv(control: &WorkerControlClient, model_instance_id: &str) -> Result<InitPagedKv> {
    loop {
        match control.recv_scheduler_message()? {
            SchedulerControlMessage::InitPagedKv(init) if init.model_instance_id == model_instance_id => {
                return Ok(init);
            }
            SchedulerControlMessage::InitPagedKv(init) => {
                tracing::warn!(
                    "Ignoring InitPagedKv for model_instance_id={} while waiting for {}",
                    init.model_instance_id,
                    model_instance_id,
                );
            }
            SchedulerControlMessage::Hello(hello) => {
                tracing::debug!(
                    "SchedulerHello while waiting InitPagedKv: protocol={} heartbeat_interval_ms={}",
                    hello.protocol_version,
                    hello.heartbeat_interval_ms,
                );
            }
            SchedulerControlMessage::LoadModel(_) => {
                tracing::warn!("Ignoring duplicate LoadModel while waiting InitPagedKv");
            }
            SchedulerControlMessage::GrantBlocks(grant) => {
                tracing::debug!("Ignoring GrantBlocks before Running: sequence_id={}", grant.sequence_id);
            }
            SchedulerControlMessage::GrantBlocksDenied(denied) => {
                tracing::debug!("Ignoring GrantBlocksDenied before Running: sequence_id={} reason={:?}", denied.sequence_id, denied.reason);
            }
        }
    }
}

fn device_mem_info(device: DeviceType) -> Result<(u64, u64)> {
    match device {
        DeviceType::Cpu => Ok((0, 0)),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(id) => {
            infer_worker::cuda::device::set_current_device(id)?;
            let (free, total) = infer_worker::cuda::device::mem_get_info()?;
            Ok((free as u64, total as u64))
        }
        #[cfg(not(feature = "cuda"))]
        DeviceType::Cuda(_) => anyhow::bail!("cuda device requested but cuda feature is disabled"),
    }
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn cli_load_model(args: &Args) -> Option<LoadModel> {
    args.model.clone().map(|model_path| LoadModel {
        model_instance_id: "default".to_string(),
        model_path,
        model_type: args.model_type.clone(),
        device: args.device.clone(),
        max_batch_tokens: args.max_batch_tokens,
        max_batch_seqs: args.max_batch_seqs,
        max_model_len: 0,
        mem_fraction_static: 1.0,
        tp_rank: 0,
        tp_size: 1,
        pp_rank: 0,
        pp_size: 1,
        kv_cache_mode: Some("slot".to_string()),
        kv_cache_memory_fraction: Some(0.95),
    })
}

fn wait_for_load_model(control: &WorkerControlClient) -> Result<LoadModel> {
    control.send_progress(WorkerState::Registered, "waiting for LoadModel")?;
    loop {
        match control.recv_scheduler_message()? {
            SchedulerControlMessage::Hello(hello) => {
                tracing::info!(
                    "SchedulerHello: protocol={} heartbeat_interval_ms={}",
                    hello.protocol_version,
                    hello.heartbeat_interval_ms,
                );
            }
            SchedulerControlMessage::LoadModel(cmd) => {
                tracing::info!(
                    "LoadModel received: model_instance_id={} model_type={} path={} max_batch_tokens={} max_batch_seqs={}",
                    cmd.model_instance_id,
                    cmd.model_type,
                    cmd.model_path,
                    cmd.max_batch_tokens,
                    cmd.max_batch_seqs,
                );
                return Ok(cmd);
            }
            SchedulerControlMessage::InitPagedKv(init) => {
                tracing::debug!(
                    "Ignoring InitPagedKv before LoadModel: model_instance_id={} blocks={}/{} block_size={}",
                    init.model_instance_id,
                    init.initial_num_blocks,
                    init.max_num_blocks,
                    init.block_size,
                );
            }
            SchedulerControlMessage::GrantBlocks(grant) => {
                tracing::debug!(
                    "Ignoring GrantBlocks before LoadModel: sequence_id={} blocks={}",
                    grant.sequence_id,
                    grant.block_ids.len(),
                );
            }
            SchedulerControlMessage::GrantBlocksDenied(denied) => {
                tracing::debug!(
                    "Ignoring GrantBlocksDenied before LoadModel: sequence_id={} reason={:?}",
                    denied.sequence_id,
                    denied.reason,
                );
            }
        }
    }
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

fn default_worker_id(device: &str) -> String {
    let safe_device = device.replace([':', '/'], "_");
    format!("worker-{}-{}", std::process::id(), safe_device)
}

fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
}
