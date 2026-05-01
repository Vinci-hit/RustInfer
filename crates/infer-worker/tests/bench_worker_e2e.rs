//! 端到端吞吐 Bench：通过 WorkerServer + ModelRunner 经 ZMQ 驱动，
//! 对比直接调用 `forward_batch_decode` 的吞吐。
//!
//! 用法:
//!   cargo test --release --package infer-worker --test bench_worker_e2e \
//!       --features cuda -- --ignored --nocapture --test-threads=1

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use infer_worker::base::{DataType, DeviceType};
use infer_worker::model::llm::llama3::Llama3;
use infer_worker::tensor::Tensor;
use infer_worker::worker::{
    BatchWorkspace, ModelRunner, SharedBuffers, WorkerServer,
};
use infer_worker::worker::protocol::*;

const MODEL_PATH: &str = "/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b";

fn maybe_skip() -> bool {
    if !std::path::Path::new(MODEL_PATH).join("config.json").exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return true;
    }
    false
}

/// Worker 端到端 bench：从调度器侧 push B 个 prefill 请求 → 等 Runner 推完 N 步 decode → 统计时间。
#[test]
#[ignore = "worker E2E benchmark"]
fn bench_worker_e2e_vs_direct() {
    if maybe_skip() {
        return;
    }
    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let device = DeviceType::Cuda(0);
    let prompt = "The quick brown fox jumps over the lazy dog"; // ~10 tokens
    let warmup_steps: usize = 50;
    let bench_steps: usize = 400;

    let model_template = Llama3::new(MODEL_PATH, device).expect("load model");
    let prompt_tokens: Vec<i32> = model_template.tokenizer().encode(prompt).expect("tokenize");
    let prompt_len = prompt_tokens.len();
    let head_num = model_template.config().head_num;
    let head_size = model_template.config().head_size;
    drop(model_template);

    eprintln!(
        "\n==== bench_worker_e2e_vs_direct (Llama-3.2-1B, CUDA, BF16)\n\
         prompt_tokens = {}\nwarmup = {} steps\nbench = {} steps",
        prompt_len, warmup_steps, bench_steps,
    );

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Path A: 直调 forward_batch_decode (跟 test_batch_forward 一致) ║
    // ╚═══════════════════════════════════════════════════════════╝
    fn run_direct(batch_size: usize, prompt_tokens: &[i32], warmup: usize, bench: usize) -> (f64, f64) {
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(MODEL_PATH, device).expect("load model");
        let prompt_len = prompt_tokens.len();

        let mut states: Vec<infer_worker::model::runtime::InferenceState> =
            (0..batch_size).map(|_| model.create_state().unwrap()).collect();

        for state in states.iter_mut() {
            let mut input_tokens_t = Tensor::new(&[prompt_len], DataType::I32, DeviceType::Cpu).unwrap();
            input_tokens_t.as_i32_mut().unwrap().as_slice_mut().unwrap()
                .copy_from_slice(prompt_tokens);
            let mut pos_t = Tensor::new(&[1], DataType::I32, DeviceType::Cpu).unwrap();
            pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
            let _ = model.forward_prefill(state, &input_tokens_t, &pos_t, prompt_len).unwrap();
        }

        let mut workspace = BatchWorkspace::new(model.config(), 64, batch_size, device).unwrap();
        workspace.sin_cache.copy_from(
            states[0].workspace.get(&infer_worker::model::BufferType::SinCache).unwrap()
        ).unwrap();
        workspace.cos_cache.copy_from(
            states[0].workspace.get(&infer_worker::model::BufferType::CosCache).unwrap()
        ).unwrap();

        let cuda_cfg = infer_worker::cuda::CudaConfig::new()
            .and_then(|c| c.with_flash_decode(
                model.config().head_num, model.config().head_size, batch_size,
            )).expect("cuda cfg");

        for step in 0..warmup {
            let positions: Vec<i32> = (0..batch_size).map(|_| (prompt_len + step) as i32).collect();
            let mut refs: Vec<&mut _> = states.iter_mut().collect();
            let _ = model.forward_batch_decode(
                refs.as_mut_slice(), &mut workspace, &positions, Some(&cuda_cfg),
            ).unwrap();
        }
        cuda_cfg.sync_stream().unwrap();

        let start = Instant::now();
        for step in 0..bench {
            let positions: Vec<i32> = (0..batch_size)
                .map(|_| (prompt_len + warmup + step) as i32).collect();
            let mut refs: Vec<&mut _> = states.iter_mut().collect();
            let _ = model.forward_batch_decode(
                refs.as_mut_slice(), &mut workspace, &positions, Some(&cuda_cfg),
            ).unwrap();
        }
        cuda_cfg.sync_stream().unwrap();
        let elapsed = start.elapsed();

        let per_step_us = elapsed.as_secs_f64() * 1e6 / bench as f64;
        let tok_per_sec = (batch_size as f64 * bench as f64) / elapsed.as_secs_f64();
        (per_step_us, tok_per_sec)
    }

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Path B: 通过 WorkerServer + ModelRunner 全链路 ZMQ 往返    ║
    // ╚═══════════════════════════════════════════════════════════╝
    fn run_worker(batch_size: usize, prompt_tokens: &[i32], warmup: usize, bench: usize) -> (f64, f64) {
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(MODEL_PATH, device).expect("load model");

        let max_num_seqs = batch_size.max(1);
        let states: Vec<_> = (0..max_num_seqs).map(|_| model.create_state().unwrap()).collect();

        let shared = SharedBuffers::new(2048, max_num_seqs, device).unwrap();

        // ZMQ endpoints 用 inproc（同进程内）；唯一化避免多次 run_worker 冲突
        let tag = format!(
            "{}-{}", batch_size, std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );
        let ep_in = format!("inproc://bench-worker-in-{}", tag);
        let ep_out = format!("inproc://bench-worker-out-{}", tag);

        let zmq_ctx = zmq::Context::new();
        let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        worker_pull.bind(&ep_in).unwrap();
        let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        worker_push.bind(&ep_out).unwrap();
        let scheduler_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        scheduler_push.connect(&ep_in).unwrap();
        let scheduler_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        scheduler_pull.connect(&ep_out).unwrap();
        scheduler_pull.set_rcvtimeo(300_000).unwrap();

        let runner_shared = Arc::clone(&shared);
        let runner = ModelRunner::new(model, states, runner_shared, 0).expect("runner");
        thread::Builder::new().name(format!("runner-b{}", batch_size))
            .spawn(move || runner.run()).unwrap();

        let server_shared = Arc::clone(&shared);
        // EOS = -1 禁用 EOS 判定（贪心采样下真实模型一定会在某步出 128009，bench 场景要屏蔽）。
        let server = WorkerServer::new(worker_pull, worker_push, server_shared, 0, -1);
        thread::Builder::new().name(format!("server-b{}", batch_size))
            .spawn(move || server.run()).unwrap();

        // 等 runner/server 就绪
        thread::sleep(Duration::from_millis(100));

        // 发送 B 个 prefill 请求（一次性塞进 PrefillBatchCmd）
        let total_tokens: usize = prompt_tokens.len() * batch_size;
        let max_tokens_per_req = warmup + bench + 16;

        let mut input_ids = Vec::with_capacity(total_tokens);
        let mut q_start_loc: Vec<u32> = Vec::with_capacity(batch_size);
        let mut num_computed_tokens: Vec<u32> = Vec::with_capacity(batch_size);
        let mut kv_slots: Vec<u32> = Vec::with_capacity(batch_size);
        let mut sampling_params: Vec<SamplingParams> = Vec::with_capacity(batch_size);
        let mut request_metas: Vec<RequestMeta> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            q_start_loc.push(input_ids.len() as u32);
            input_ids.extend_from_slice(prompt_tokens);
            num_computed_tokens.push(0);
            kv_slots.push(i as u32);
            sampling_params.push(SamplingParams { temperature: 0.0, top_p: 1.0, top_k: -1 });
            request_metas.push(RequestMeta {
                request_id: format!("bench-{}-{}", batch_size, i),
                max_tokens: max_tokens_per_req,
            });
        }

        let cmd = PrefillBatchCmd {
            input_ids,
            q_start_loc,
            num_computed_tokens,
            kv_slots,
            sampling_params,
            request_metas,
        };
        scheduler_push.send(&rmp_serde::to_vec(&cmd).unwrap(), 0).unwrap();

        // 收第 1 步（prefill 的输出）——它对应 prompt 的 first token
        let _ = scheduler_pull.recv_bytes(0).expect("prefill output");

        // 再收 warmup 步
        for _ in 0..warmup {
            let _ = scheduler_pull.recv_bytes(0).expect("warmup step");
        }

        // ═══ Bench window ═══
        let start = Instant::now();
        for _ in 0..bench {
            let _ = scheduler_pull.recv_bytes(0).expect("bench step");
        }
        let elapsed = start.elapsed();

        let per_step_us = elapsed.as_secs_f64() * 1e6 / bench as f64;
        let tok_per_sec = (batch_size as f64 * bench as f64) / elapsed.as_secs_f64();
        (per_step_us, tok_per_sec)
    }

    // ═══ 运行 ═══
    let skip_direct = std::env::var("SKIP_DIRECT").is_ok();
    let skip_worker = std::env::var("SKIP_WORKER").is_ok();
    let only_b = std::env::var("ONLY_B").ok().and_then(|s| s.parse::<usize>().ok());

    let mut d_tok_b1 = 0.0;
    let mut d_tok_b2 = 0.0;
    if !skip_direct {
        eprintln!("\n-- Path A: direct forward_batch_decode --");
        if only_b != Some(2) {
            let (us, tok) = run_direct(1, &prompt_tokens, warmup_steps, bench_steps);
            eprintln!("  B=1: per-step = {:>7.2} us  throughput = {:>7.1} tok/s", us, tok);
            d_tok_b1 = tok;
        }
        if only_b != Some(1) {
            let (us, tok) = run_direct(2, &prompt_tokens, warmup_steps, bench_steps);
            eprintln!("  B=2: per-step = {:>7.2} us  throughput = {:>7.1} tok/s", us, tok);
            d_tok_b2 = tok;
        }
    }

    let mut w_tok_b1 = 0.0;
    let mut w_tok_b2 = 0.0;
    if !skip_worker {
        eprintln!("\n-- Path B: worker E2E (ZMQ + Server + Runner) --");
        if only_b != Some(2) {
            let (us, tok) = run_worker(1, &prompt_tokens, warmup_steps, bench_steps);
            eprintln!("  B=1: per-step = {:>7.2} us  throughput = {:>7.1} tok/s", us, tok);
            w_tok_b1 = tok;
        }
        if only_b != Some(1) {
            let (us, tok) = run_worker(2, &prompt_tokens, warmup_steps, bench_steps);
            eprintln!("  B=2: per-step = {:>7.2} us  throughput = {:>7.1} tok/s", us, tok);
            w_tok_b2 = tok;
        }
    }

    eprintln!("\n---- Summary (worker overhead vs direct) ----");
    if d_tok_b1 > 0.0 && w_tok_b1 > 0.0 {
        eprintln!(
            "B=1:  direct = {:>7.1} tok/s,  worker = {:>7.1} tok/s  ({:.3}x)",
            d_tok_b1, w_tok_b1, w_tok_b1 / d_tok_b1,
        );
    }
    if d_tok_b2 > 0.0 && w_tok_b2 > 0.0 {
        eprintln!(
            "B=2:  direct = {:>7.1} tok/s,  worker = {:>7.1} tok/s  ({:.3}x)",
            d_tok_b2, w_tok_b2, w_tok_b2 / d_tok_b2,
        );
    }

    // 让所有线程 drop 时退出（ModelRunner/WorkerServer 目前是 loop forever，
    // 主进程通过 exit 结束整个 test harness）
    let _ = (head_num, head_size);
    std::process::exit(0);
}
