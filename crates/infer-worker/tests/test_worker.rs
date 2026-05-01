//! 集成测试: 模拟调度器向 Worker 发送 prefill 指令, 验证能正确收到输出。
//!
//! - test_worker_pipeline_cpu / test_worker_multi_request_batch:
//!   使用 DummyModelRunner (CPU, 无需 GPU/模型)
//!
//! - test_worker_with_llama3:
//!   加载真实 Llama3-1B 模型, CUDA 推理

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use infer_worker::base::DeviceType;
use infer_worker::worker::protocol::*;
use infer_worker::worker::shared_buffers::SharedBuffers;
use infer_worker::worker::WorkerServer;
use infer_worker::worker::runner_dummy::DummyModelRunner;

/// 使用 CPU device 跑通整个流水线 (无需 GPU)
#[test]
fn test_worker_pipeline_cpu() {
    // 使用 inproc transport 避免文件系统 IPC 清理问题
    let zmq_in_endpoint = "inproc://test-worker-in";
    let zmq_out_endpoint = "inproc://test-worker-out";

    let zmq_ctx = zmq::Context::new();

    // Worker 侧 sockets
    let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    worker_pull.bind(zmq_in_endpoint).unwrap();

    let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    worker_push.bind(zmq_out_endpoint).unwrap();

    // 调度器侧 sockets
    let scheduler_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    scheduler_push.connect(zmq_in_endpoint).unwrap();

    let scheduler_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    scheduler_pull.connect(zmq_out_endpoint).unwrap();

    // 设置 timeout 避免测试卡死
    scheduler_pull.set_rcvtimeo(5000).unwrap(); // 5s timeout

    // 共享 buffer (CPU device)
    let device = DeviceType::Cpu;
    let shared = SharedBuffers::new(2048, 64, device).unwrap();

    // EOS token = 128009 (llama3), 我们用 dummy token 42, 不会触发 EOS
    // 但 max_tokens = 3, 所以第 3 步会 finished=true
    let eos_token_id = 128009;

    // 启动 Runner 线程
    let runner_shared = Arc::clone(&shared);
    let runner = DummyModelRunner::new(runner_shared);
    let runner_handle = thread::spawn(move || runner.run());

    // 启动 Server 线程
    let server_shared = Arc::clone(&shared);
    let server = WorkerServer::new(worker_pull, worker_push, server_shared, 0, eos_token_id);
    let server_handle = thread::spawn(move || server.run());

    // 等一下让线程启动
    thread::sleep(Duration::from_millis(100));

    // ══ 模拟调度器: 发送一个 prefill 请求 ══
    let cmd = PrefillBatchCmd {
        input_ids: vec![1, 2, 3, 4, 5], // 5 tokens
        q_start_loc: vec![0],            // 1 个请求, 从 offset 0 开始
        num_computed_tokens: vec![0],    // 首次 prefill
        kv_slots: vec![0],
        sampling_params: vec![SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
        }],
        request_metas: vec![RequestMeta {
            request_id: "req-001".to_string(),
            max_tokens: 3,
        }],
    };

    let data = rmp_serde::to_vec(&cmd).unwrap();
    scheduler_push.send(&data, 0).unwrap();

    // ══ 接收 step outputs ══
    // 因为 max_tokens=3, 应该收到 3 次 StepOutput (每次一个 token)
    // 第 3 次 finished=true

    for step in 0..3 {
        let resp_data = scheduler_pull
            .recv_bytes(0)
            .expect(&format!("Timeout waiting for step {} output", step));
        let output: StepOutput = rmp_serde::from_slice(&resp_data)
            .expect(&format!("Failed to deserialize step {} output", step));

        assert_eq!(output.tokens.len(), 1, "Step {} should have 1 token", step);
        assert_eq!(output.tokens[0].request_id, "req-001");
        assert_eq!(output.tokens[0].token_id, 42, "Dummy runner outputs 42");

        if step == 2 {
            // 第 3 步: generated_count=3 >= max_tokens=3 → finished
            assert!(
                output.tokens[0].finished,
                "Step 2 should be finished (max_tokens reached)"
            );
        } else {
            assert!(
                !output.tokens[0].finished,
                "Step {} should not be finished",
                step
            );
        }
    }

    // 测试通过, 清理线程 (drop scheduler sockets 会导致 Worker 最终 panic/exit)
    // 这里直接结束测试, 不等线程 (它们会在 ZMQ 错误时 panic)
    drop(scheduler_push);
    drop(scheduler_pull);

    // 不 join (会死锁因为 server 在等新 prefill), 测试目的已达到
    drop(runner_handle);
    drop(server_handle);
}

/// 测试多个请求 batch
#[test]
fn test_worker_multi_request_batch() {
    let zmq_ctx = zmq::Context::new();

    let zmq_in_endpoint = "inproc://test-worker-multi-in";
    let zmq_out_endpoint = "inproc://test-worker-multi-out";

    let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    worker_pull.bind(zmq_in_endpoint).unwrap();
    let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    worker_push.bind(zmq_out_endpoint).unwrap();

    let scheduler_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    scheduler_push.connect(zmq_in_endpoint).unwrap();
    let scheduler_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    scheduler_pull.connect(zmq_out_endpoint).unwrap();
    scheduler_pull.set_rcvtimeo(5000).unwrap();

    let device = DeviceType::Cpu;
    let shared = SharedBuffers::new(2048, 64, device).unwrap();
    let eos_token_id = 128009;

    let runner_shared = Arc::clone(&shared);
    let runner = DummyModelRunner::new(runner_shared);
    thread::spawn(move || runner.run());

    let server_shared = Arc::clone(&shared);
    let server = WorkerServer::new(worker_pull, worker_push, server_shared, 0, eos_token_id);
    thread::spawn(move || server.run());

    thread::sleep(Duration::from_millis(100));

    // 发送 2 个请求: req-A (3 tokens prompt, max=2), req-B (2 tokens prompt, max=1)
    let cmd = PrefillBatchCmd {
        input_ids: vec![10, 20, 30, 40, 50], // req-A: [10,20,30], req-B: [40,50]
        q_start_loc: vec![0, 3],              // req-A 从 0, req-B 从 3
        num_computed_tokens: vec![0, 0],
        kv_slots: vec![0, 1],
        sampling_params: vec![
            SamplingParams { temperature: 1.0, top_p: 0.9, top_k: 50 },
            SamplingParams { temperature: 1.0, top_p: 0.9, top_k: 50 },
        ],
        request_metas: vec![
            RequestMeta { request_id: "req-A".to_string(), max_tokens: 2 },
            RequestMeta { request_id: "req-B".to_string(), max_tokens: 1 },
        ],
    };

    let data = rmp_serde::to_vec(&cmd).unwrap();
    scheduler_push.send(&data, 0).unwrap();

    // Step 0: 两个请求都输出
    // req-B: max_tokens=1, generated_count=1 → finished
    // req-A: max_tokens=2, generated_count=1 → not finished
    let resp = scheduler_pull.recv_bytes(0).expect("Timeout step 0");
    let output: StepOutput = rmp_serde::from_slice(&resp).unwrap();
    assert_eq!(output.tokens.len(), 2);

    // 找到各自的 token
    let tok_a = output.tokens.iter().find(|t| t.request_id == "req-A").unwrap();
    let tok_b = output.tokens.iter().find(|t| t.request_id == "req-B").unwrap();
    assert_eq!(tok_a.token_id, 42);
    assert!(!tok_a.finished);
    assert_eq!(tok_b.token_id, 42);
    assert!(tok_b.finished, "req-B should finish at step 0 (max_tokens=1)");

    // Step 1: 只有 req-A 还在
    let resp = scheduler_pull.recv_bytes(0).expect("Timeout step 1");
    let output: StepOutput = rmp_serde::from_slice(&resp).unwrap();
    assert_eq!(output.tokens.len(), 1);
    assert_eq!(output.tokens[0].request_id, "req-A");
    assert!(output.tokens[0].finished, "req-A should finish at step 1 (max_tokens=2)");
}

/// 加载真实 Llama3-1B 模型 (CPU)，验证 Worker 输出与 model.generate() 一致
///
/// 测试步骤:
///   1. 用 model.generate() 作为 baseline 生成 N 个 token
///   2. 用 Worker pipeline 生成 N 个 token
///   3. 逐 token 对比，确保完全一致 (greedy decoding, 结果确定)
#[test]
fn test_worker_with_llama3() {
    use infer_worker::model::llm::llama3::Llama3;
    use infer_worker::worker::ModelRunner;

    let model_path = "/data/home/vinciiliu/models/Llama-3.2-1B-Instruct";
    if !std::path::Path::new(model_path).join("config.json").exists() {
        eprintln!("Skipping test: model not found at {}", model_path);
        return;
    }

    tracing_subscriber::fmt()
        .with_env_filter("info")
        .try_init()
        .ok();

    let device = DeviceType::Cpu;
    let prompt = "1+1=";
    let max_gen = 10;
    let eos_token_id = 128009;

    // ═══ Baseline: model.generate() ═══
    tracing::info!("Loading Llama3-1B for baseline...");
    let model_baseline = Llama3::new(model_path, device).expect("load model");
    let mut state_baseline = model_baseline.create_state().expect("create state");
    let prompt_tokens = model_baseline.tokenizer().encode(prompt).expect("tokenize");
    tracing::info!("Prompt '{}' → tokens {:?}", prompt, &prompt_tokens);

    let (baseline_text, baseline_num_tokens, _, _, _) = model_baseline
        .generate(&mut state_baseline, prompt, max_gen, false)
        .expect("baseline generate");
    tracing::info!("Baseline output: '{}' ({} tokens)", baseline_text, baseline_num_tokens);

    // 重新 generate 获取 token ids (generate 只返回 text)
    // 手动做 prefill + decode 收集 token ids
    let mut state2 = model_baseline.create_state().expect("create state2");
    let mut input_tokens_t = infer_worker::tensor::Tensor::new(
        &[prompt_tokens.len()], infer_worker::base::DataType::I32, DeviceType::Cpu,
    ).unwrap();
    input_tokens_t.as_i32_mut().unwrap().as_slice_mut().unwrap()
        .copy_from_slice(&prompt_tokens);
    let mut pos_t = infer_worker::tensor::Tensor::new(
        &[1], infer_worker::base::DataType::I32, DeviceType::Cpu,
    ).unwrap();
    pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;

    let first_tok = model_baseline.forward_prefill(
        &mut state2, &input_tokens_t, &pos_t, prompt_tokens.len(),
    ).expect("baseline prefill");

    let mut baseline_tokens = vec![first_tok];
    let mut current_token = first_tok;
    let mut input_1 = infer_worker::tensor::Tensor::new(
        &[1], infer_worker::base::DataType::I32, DeviceType::Cpu,
    ).unwrap();

    for pos in prompt_tokens.len()..(prompt_tokens.len() - 1 + max_gen) {
        if current_token == eos_token_id { break; }
        pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = pos as i32;
        input_1.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = current_token;
        let next = model_baseline.forward_decoding(
            &mut state2, &input_1, &pos_t,
        ).expect("baseline decode");
        baseline_tokens.push(next);
        current_token = next;
    }
    tracing::info!("Baseline tokens: {:?}", &baseline_tokens);

    // ═══ Worker pipeline ═══
    tracing::info!("Loading Llama3-1B for Worker...");
    let model_worker = Llama3::new(model_path, device).expect("load model worker");
    let max_num_seqs = 1;
    let mut states = Vec::with_capacity(max_num_seqs);
    for _ in 0..max_num_seqs {
        states.push(model_worker.create_state().expect("create_state"));
    }

    let shared = SharedBuffers::new(2048, max_num_seqs, device).unwrap();

    let zmq_ctx = zmq::Context::new();
    let zmq_in_ep = "inproc://test-llama-in";
    let zmq_out_ep = "inproc://test-llama-out";

    let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    worker_pull.bind(zmq_in_ep).unwrap();
    let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    worker_push.bind(zmq_out_ep).unwrap();

    let scheduler_push = zmq_ctx.socket(zmq::PUSH).unwrap();
    scheduler_push.connect(zmq_in_ep).unwrap();
    let scheduler_pull = zmq_ctx.socket(zmq::PULL).unwrap();
    scheduler_pull.connect(zmq_out_ep).unwrap();
    scheduler_pull.set_rcvtimeo(60000).unwrap();

    let runner_shared = Arc::clone(&shared);
    let runner = ModelRunner::new(model_worker, states, runner_shared, 0);
    thread::spawn(move || runner.run());

    let server_shared = Arc::clone(&shared);
    let server = WorkerServer::new(worker_pull, worker_push, server_shared, 0, eos_token_id);
    thread::spawn(move || server.run());

    thread::sleep(Duration::from_millis(100));

    // 发 prefill
    let cmd = PrefillBatchCmd {
        input_ids: prompt_tokens.clone(),
        q_start_loc: vec![0],
        num_computed_tokens: vec![0],
        kv_slots: vec![0],
        sampling_params: vec![SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
        }],
        request_metas: vec![RequestMeta {
            request_id: "cmp-001".to_string(),
            max_tokens: max_gen,
        }],
    };

    scheduler_push.send(&rmp_serde::to_vec(&cmd).unwrap(), 0).unwrap();

    // 收集 worker 输出
    let mut worker_tokens: Vec<i32> = Vec::new();
    for step in 0..max_gen {
        let resp = scheduler_pull.recv_bytes(0)
            .unwrap_or_else(|_| panic!("Timeout at step {}", step));
        let output: StepOutput = rmp_serde::from_slice(&resp).unwrap();
        assert_eq!(output.tokens.len(), 1, "step {} should have 1 seq", step);

        let tok = &output.tokens[0];
        assert_eq!(tok.request_id, "cmp-001");
        worker_tokens.push(tok.token_id);

        tracing::info!("Worker step {}: token_id={}, finished={}", step, tok.token_id, tok.finished);

        if tok.finished { break; }
    }

    tracing::info!("Worker  tokens: {:?}", &worker_tokens);
    tracing::info!("Baseline tokens: {:?}", &baseline_tokens);

    // ═══ 逐 token 对比 ═══
    let cmp_len = baseline_tokens.len().min(worker_tokens.len());
    for i in 0..cmp_len {
        assert_eq!(
            worker_tokens[i], baseline_tokens[i],
            "Token mismatch at position {}: worker={}, baseline={}",
            i, worker_tokens[i], baseline_tokens[i],
        );
    }
    assert_eq!(
        worker_tokens.len(), baseline_tokens.len(),
        "Token count mismatch: worker={}, baseline={}",
        worker_tokens.len(), baseline_tokens.len(),
    );

    tracing::info!("All {} tokens match. Test PASSED.", cmp_len);

    std::process::exit(0);
}

