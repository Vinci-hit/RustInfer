#![cfg(feature = "models")]
//! 测试 worker 统一 forward 与串行/批量输出一致性

use infer_worker::base::DeviceType;
use infer_worker::model::llm::llama3::Llama3;
use infer_worker::model::llm::LlmModel;
use infer_worker::model::runtime::InferenceState;
use infer_worker::worker::runner::WorkerBatchMeta;
use infer_worker::worker::BatchWorkspace;

const MODEL_PATH: &str = "/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b";

#[cfg(feature = "cuda")]
type TestCudaConfig = infer_worker::cuda::CudaConfig;
#[cfg(not(feature = "cuda"))]
type TestCudaConfig = ();

fn make_workspace(model: &Llama3, max_batch_tokens: usize, max_seqs: usize, device: DeviceType) -> BatchWorkspace {
    let mut workspace = BatchWorkspace::new(model.config(), max_batch_tokens, max_seqs, device)
        .expect("create workspace");
    model.fill_rope_cache(&mut workspace.sin_cache, &mut workspace.cos_cache)
        .expect("fill rope cache");
    workspace
}

#[cfg(feature = "cuda")]
fn make_cuda_cfg(model: &Llama3, max_batch: usize) -> TestCudaConfig {
    infer_worker::cuda::CudaConfig::new()
        .and_then(|c| c.with_flash_decode(model.config().head_num, model.config().head_size, max_batch))
        .expect("create cuda config")
}

#[cfg(not(feature = "cuda"))]
fn make_cuda_cfg(_model: &Llama3, _max_batch: usize) -> TestCudaConfig {}

fn cfg_ref(cfg: &TestCudaConfig) -> Option<&infer_worker::OpConfig> {
    #[cfg(feature = "cuda")]
    { Some(cfg) }
    #[cfg(not(feature = "cuda"))]
    { let _ = cfg; None }
}

fn forward_prefill_unified(
    model: &Llama3,
    state: &mut InferenceState,
    workspace: &mut BatchWorkspace,
    tokens: &[i32],
    start_pos: i32,
    cuda_cfg: &TestCudaConfig,
) -> i32 {
    let positions: Vec<i32> = (0..tokens.len()).map(|i| start_pos + i as i32).collect();
    let q_start_loc = [0, tokens.len() as i32];
    let slot_indices = [0];
    workspace.input_tokens.write_from_i32_host(tokens, tokens.len()).unwrap();
    workspace.input_pos.write_from_i32_host(&positions, positions.len()).unwrap();
    workspace.kv_lens_cpu.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = start_pos;
    #[cfg(feature = "cuda")]
    {
        let src = workspace.kv_lens_cpu.slice(&[0], &[1]).unwrap();
        let mut dst = workspace.kv_lens_dev.slice(&[0], &[1]).unwrap();
        dst.copy_from(&src).unwrap();
    }
    let meta = WorkerBatchMeta {
        q_start_loc: &q_start_loc,
        slot_indices: &slot_indices,
        token_ids: tokens,
        positions: &positions,
        num_decode: 0,
        num_prefill: 1,
    };
    let mut out = infer_worker::tensor::Tensor::new(&[1], infer_worker::base::DataType::I32, DeviceType::Cuda(0)).unwrap();
    let mut refs = vec![state];
    model.forward(refs.as_mut_slice(), workspace, &meta, &mut out, cfg_ref(cuda_cfg)).unwrap();
    let tok = out.to_cpu().unwrap().as_i32().unwrap().as_slice().unwrap()[0];
    refs[0].output_token.copy_from(&out.slice(&[0], &[1]).unwrap()).unwrap();
    tok
}

fn forward_decode_unified(
    model: &Llama3,
    states: &mut [InferenceState],
    workspace: &mut BatchWorkspace,
    positions: &[i32],
    cuda_cfg: &TestCudaConfig,
) -> Vec<i32> {
    let batch = states.len();
    assert_eq!(batch, positions.len());
    let token_ids = vec![0i32; batch];
    let q_start_loc: Vec<i32> = (0..=batch).map(|i| i as i32).collect();
    let slot_indices: Vec<i32> = (0..batch).map(|i| i as i32).collect();
    for (i, state) in states.iter_mut().enumerate() {
        let mut dst = workspace.input_tokens.slice(&[i], &[1]).unwrap();
        dst.copy_from_on_current_stream(&state.output_token).unwrap();
    }
    workspace.input_pos.write_from_i32_host(positions, positions.len()).unwrap();
    {
        let kv = workspace.kv_lens_cpu.as_i32_mut().unwrap().as_slice_mut().unwrap();
        kv[..batch].copy_from_slice(positions);
    }
    #[cfg(feature = "cuda")]
    {
        let src = workspace.kv_lens_cpu.slice(&[0], &[batch]).unwrap();
        let mut dst = workspace.kv_lens_dev.slice(&[0], &[batch]).unwrap();
        dst.copy_from(&src).unwrap();
    }
    let meta = WorkerBatchMeta {
        q_start_loc: &q_start_loc,
        slot_indices: &slot_indices,
        token_ids: &token_ids,
        positions,
        num_decode: batch,
        num_prefill: 0,
    };
    let mut out = infer_worker::tensor::Tensor::new(&[batch], infer_worker::base::DataType::I32, DeviceType::Cuda(0)).unwrap();
    let mut refs: Vec<&mut _> = states.iter_mut().collect();
    model.forward(refs.as_mut_slice(), workspace, &meta, &mut out, cfg_ref(cuda_cfg)).unwrap();
    for i in 0..batch {
        refs[i].output_token.copy_from(&out.slice(&[i], &[1]).unwrap()).unwrap();
    }
    out.to_cpu().unwrap().as_i32().unwrap().as_slice().unwrap().to_vec()
}

/// 对比统一 forward 的 batch_size=1 decode 与同一接口串行输出是否完全一致
#[test]
fn test_batch_decode_matches_serial() {
    if !std::path::Path::new(MODEL_PATH).join("config.json").exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let device = DeviceType::Cuda(0);
    let prompt = "1+1=";
    let num_decode_steps = 5;

    let model = Llama3::new(MODEL_PATH, device).expect("load model");
    let prompt_tokens = model.tokenizer().encode(prompt).expect("tokenize");
    tracing::info!("Prompt '{}' → {:?}", prompt, &prompt_tokens);

    // ═══ 串行 baseline（也走统一 worker forward，B=1） ═══
    let mut state_serial = model.create_state().expect("create state");
    let mut workspace_serial = make_workspace(&model, 64, 4, device);
    let cuda_serial = make_cuda_cfg(&model, 1);
    let first_tok = forward_prefill_unified(
        &model, &mut state_serial, &mut workspace_serial, &prompt_tokens, 0, &cuda_serial,
    );

    let mut serial_tokens = vec![first_tok];
    for step in 0..num_decode_steps {
        let pos = (prompt_tokens.len() + step) as i32;
        let tok = forward_decode_unified(
            &model, std::slice::from_mut(&mut state_serial), &mut workspace_serial, &[pos], &cuda_serial,
        )[0];
        serial_tokens.push(tok);
    }
    tracing::info!("Serial tokens: {:?}", &serial_tokens);

    // ═══ Batch decode (batch_size=1) ═══
    let mut state_batch = model.create_state().expect("create state");
    let mut workspace_batch = make_workspace(&model, 64, 4, device);
    let cuda_batch = make_cuda_cfg(&model, 1);
    let first_tok_batch = forward_prefill_unified(
        &model, &mut state_batch, &mut workspace_batch, &prompt_tokens, 0, &cuda_batch,
    );
    assert_eq!(first_tok_batch, first_tok, "Prefill output should match");

    let mut batch_tokens = vec![first_tok_batch];
    let mut states = [state_batch];
    for step in 0..num_decode_steps {
        let pos = (prompt_tokens.len() + step) as i32;
        let result = forward_decode_unified(&model, &mut states, &mut workspace_batch, &[pos], &cuda_batch);
        assert_eq!(result.len(), 1);
        batch_tokens.push(result[0]);
    }
    tracing::info!("Batch  tokens: {:?}", &batch_tokens);

    assert_eq!(serial_tokens, batch_tokens);
    tracing::info!("All {} tokens match!", serial_tokens.len());
}

/// 对比 batch_size=2 的 batch decode: 两个不同 prompt 各自输出是否与串行一致
#[test]
fn test_batch_decode_two_seqs() {
    if !std::path::Path::new(MODEL_PATH).join("config.json").exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let device = DeviceType::Cuda(0);
    let prompt_a = "1+1=";
    let prompt_b = "Hello";
    let num_decode_steps = 3;

    let model = Llama3::new(MODEL_PATH, device).expect("load model");
    let tokens_a = model.tokenizer().encode(prompt_a).unwrap();
    let tokens_b = model.tokenizer().encode(prompt_b).unwrap();

    // ═══ 串行 baseline for prompt A ═══
    let mut state_a_serial = model.create_state().unwrap();
    let mut ws_a_serial = make_workspace(&model, 64, 4, device);
    let cuda_a = make_cuda_cfg(&model, 1);
    let first_a = forward_prefill_unified(&model, &mut state_a_serial, &mut ws_a_serial, &tokens_a, 0, &cuda_a);
    let mut serial_a = vec![first_a];
    for step in 0..num_decode_steps {
        let p = (tokens_a.len() + step) as i32;
        serial_a.push(forward_decode_unified(&model, std::slice::from_mut(&mut state_a_serial), &mut ws_a_serial, &[p], &cuda_a)[0]);
    }

    // ═══ 串行 baseline for prompt B ═══
    let mut state_b_serial = model.create_state().unwrap();
    let mut ws_b_serial = make_workspace(&model, 64, 4, device);
    let cuda_b = make_cuda_cfg(&model, 1);
    let first_b = forward_prefill_unified(&model, &mut state_b_serial, &mut ws_b_serial, &tokens_b, 0, &cuda_b);
    let mut serial_b = vec![first_b];
    for step in 0..num_decode_steps {
        let p = (tokens_b.len() + step) as i32;
        serial_b.push(forward_decode_unified(&model, std::slice::from_mut(&mut state_b_serial), &mut ws_b_serial, &[p], &cuda_b)[0]);
    }

    tracing::info!("Serial A: {:?}", &serial_a);
    tracing::info!("Serial B: {:?}", &serial_b);

    // ═══ Batch decode (2 seqs) ═══
    let mut state_a = model.create_state().unwrap();
    let mut state_b = model.create_state().unwrap();
    let mut workspace = make_workspace(&model, 64, 4, device);
    let cuda_cfg = make_cuda_cfg(&model, 2);

    let first_a2 = forward_prefill_unified(&model, &mut state_a, &mut workspace, &tokens_a, 0, &cuda_cfg);
    let first_b2 = forward_prefill_unified(&model, &mut state_b, &mut workspace, &tokens_b, 0, &cuda_cfg);
    assert_eq!(first_a2, first_a);
    assert_eq!(first_b2, first_b);

    let mut batch_a = vec![first_a2];
    let mut batch_b = vec![first_b2];
    let mut states = [state_a, state_b];

    for step in 0..num_decode_steps {
        let positions = [
            (tokens_a.len() + step) as i32,
            (tokens_b.len() + step) as i32,
        ];
        let result = forward_decode_unified(&model, &mut states, &mut workspace, &positions, &cuda_cfg);
        assert_eq!(result.len(), 2);
        batch_a.push(result[0]);
        batch_b.push(result[1]);
    }

    tracing::info!("Batch  A: {:?}", &batch_a);
    tracing::info!("Batch  B: {:?}", &batch_b);
    assert_eq!(serial_a, batch_a);
    assert_eq!(serial_b, batch_b);
    tracing::info!("All tokens match for both seqs!");
}

// ============================================================================
// Bench: B=1 vs B=2 吞吐对比
//   用法:  cargo test --release --package infer-worker --test test_batch_forward \
//              --features cuda -- --ignored --nocapture --test-threads=1 bench_batch_throughput
// ============================================================================
#[test]
#[ignore = "benchmark, wall-clock only"]
fn bench_batch_throughput() {
    if !std::path::Path::new(MODEL_PATH).join("config.json").exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }
    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let device = DeviceType::Cuda(0);
    let prompt = "The quick brown fox jumps over the lazy dog";
    let warmup_steps: usize = std::env::var("BENCH_WARMUP").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(50);
    let bench_steps: usize  = std::env::var("BENCH_STEPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(400);

    let model = Llama3::new(MODEL_PATH, device).expect("load model");
    let prompt_tokens = model.tokenizer().encode(prompt).expect("tokenize");
    let prompt_len = prompt_tokens.len();
    eprintln!(
        "\n==== bench_batch_throughput (Llama-3.2-1B, CUDA, BF16)\nprompt_tokens = {}\nwarmup={} steps, bench={} steps",
        prompt_len, warmup_steps, bench_steps,
    );

    let run_one = |batch_size: usize| -> (f64, f64) {
        let mut states: Vec<InferenceState> =
            (0..batch_size).map(|_| model.create_state().unwrap()).collect();
        let mut workspace = make_workspace(&model, 64, batch_size, device);
        let cuda_cfg = make_cuda_cfg(&model, batch_size);

        for state in states.iter_mut() {
            let _first = forward_prefill_unified(&model, state, &mut workspace, &prompt_tokens, 0, &cuda_cfg);
        }

        for step in 0..warmup_steps {
            let positions: Vec<i32> = (0..batch_size).map(|_| (prompt_len + step) as i32).collect();
            let _ = forward_decode_unified(&model, &mut states, &mut workspace, &positions, &cuda_cfg);
        }
        #[cfg(feature = "cuda")]
        cuda_cfg.sync_stream().unwrap();

        let start = std::time::Instant::now();
        for step in 0..bench_steps {
            let positions: Vec<i32> = (0..batch_size)
                .map(|_| (prompt_len + warmup_steps + step) as i32)
                .collect();
            let _ = forward_decode_unified(&model, &mut states, &mut workspace, &positions, &cuda_cfg);
        }
        #[cfg(feature = "cuda")]
        cuda_cfg.sync_stream().unwrap();
        let elapsed = start.elapsed();

        let per_step_us = elapsed.as_secs_f64() * 1e6 / bench_steps as f64;
        let tokens_per_sec = (batch_size as f64 * bench_steps as f64) / elapsed.as_secs_f64();
        eprintln!(
            "  B={}: total = {:>7.2} ms | per-step = {:>8.2} us | throughput = {:>8.1} tok/s",
            batch_size, elapsed.as_secs_f64() * 1e3, per_step_us, tokens_per_sec,
        );
        (per_step_us, tokens_per_sec)
    };

    let only = std::env::var("BENCH_ONLY").unwrap_or_else(|_| "all".to_string());
    let (us_b1, tok_b1) = if only == "all" || only == "b1" { run_one(1) } else { (0.0, 0.0) };
    let (us_b2, tok_b2) = if only == "all" || only == "b2" { run_one(2) } else { (0.0, 0.0) };

    eprintln!("\n---- Summary ----");
    eprintln!(
        "unified forward B=1  per-step {:>8.2} us, throughput {:>8.1} tok/s",
        us_b1, tok_b1,
    );
    eprintln!(
        "unified forward B=2  per-step {:>8.2} us, throughput {:>8.1} tok/s",
        us_b2, tok_b2,
    );
}
