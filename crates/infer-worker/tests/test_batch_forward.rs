//! 测试 batch forward 与串行 forward 输出一致性

use infer_worker::base::DeviceType;
use infer_worker::base::DataType;
use infer_worker::model::llm::llama3::Llama3;
use infer_worker::tensor::Tensor;
use infer_worker::worker::BatchWorkspace;

const MODEL_PATH: &str = "/data/home/vinciiliu/models/Llama-3.2-1B-Instruct";

/// 对比 forward_batch_decode (batch_size=1) 与 forward_decoding 输出是否完全一致
#[test]
fn test_batch_decode_matches_serial() {
    if !std::path::Path::new(MODEL_PATH).join("config.json").exists() {
        eprintln!("Skipping: model not found at {}", MODEL_PATH);
        return;
    }

    tracing_subscriber::fmt().with_env_filter("info").try_init().ok();

    let device = DeviceType::Cpu;
    let prompt = "1+1=";
    let num_decode_steps = 5;

    // ═══ 串行 baseline ═══
    let model = Llama3::new(MODEL_PATH, device).expect("load model");
    let mut state_serial = model.create_state().expect("create state");
    let prompt_tokens = model.tokenizer().encode(prompt).expect("tokenize");
    tracing::info!("Prompt '{}' → {:?}", prompt, &prompt_tokens);

    // Prefill
    let mut input_tokens_t = Tensor::new(&[prompt_tokens.len()], DataType::I32, DeviceType::Cpu).unwrap();
    input_tokens_t.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&prompt_tokens);
    let mut pos_t = Tensor::new(&[1], DataType::I32, DeviceType::Cpu).unwrap();
    pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_tok = model.forward_prefill(&mut state_serial, &input_tokens_t, &pos_t, prompt_tokens.len()).unwrap();

    // Decode
    let mut serial_tokens = vec![first_tok];
    let mut input_1 = Tensor::new(&[1], DataType::I32, DeviceType::Cpu).unwrap();
    for step in 0..num_decode_steps {
        let pos = (prompt_tokens.len() + step) as i32;
        pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = pos;
        input_1.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = serial_tokens.last().copied().unwrap();
        let tok = model.forward_decoding(&mut state_serial, &input_1, &pos_t).unwrap();
        serial_tokens.push(tok);
    }
    tracing::info!("Serial tokens: {:?}", &serial_tokens);

    // ═══ Batch decode (batch_size=1) ═══
    let mut state_batch = model.create_state().expect("create state");

    // 先做 prefill (用串行接口)
    let mut input_tokens_t2 = Tensor::new(&[prompt_tokens.len()], DataType::I32, DeviceType::Cpu).unwrap();
    input_tokens_t2.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&prompt_tokens);
    pos_t.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_tok_batch = model.forward_prefill(&mut state_batch, &input_tokens_t2, &pos_t, prompt_tokens.len()).unwrap();
    assert_eq!(first_tok_batch, first_tok, "Prefill output should match");

    // 创建 batch workspace
    let mut workspace = BatchWorkspace::new(model.config(), 64, 4, device).expect("create workspace");

    // 复制 sin/cos cache 从 state 到 workspace
    workspace.sin_cache.copy_from(state_batch.workspace.get(&infer_worker::model::BufferType::SinCache).unwrap()).unwrap();
    workspace.cos_cache.copy_from(state_batch.workspace.get(&infer_worker::model::BufferType::CosCache).unwrap()).unwrap();

    // Batch decode
    let mut batch_tokens = vec![first_tok_batch];
    let mut states = [state_batch];
    for step in 0..num_decode_steps {
        let pos = (prompt_tokens.len() + step) as i32;
        let positions = [pos];
        let result = model.forward_batch_decode(&mut states, &mut workspace, &positions, None).unwrap();
        assert_eq!(result.len(), 1);
        batch_tokens.push(result[0]);
    }
    tracing::info!("Batch  tokens: {:?}", &batch_tokens);

    // ═══ 逐 token 对比 ═══
    assert_eq!(serial_tokens.len(), batch_tokens.len());
    for i in 0..serial_tokens.len() {
        assert_eq!(
            serial_tokens[i], batch_tokens[i],
            "Token mismatch at pos {}: serial={}, batch={}",
            i, serial_tokens[i], batch_tokens[i],
        );
    }
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

    let device = DeviceType::Cpu;
    let prompt_a = "1+1=";
    let prompt_b = "Hello";
    let num_decode_steps = 3;

    let model = Llama3::new(MODEL_PATH, device).expect("load model");

    // ═══ 串行 baseline for prompt A ═══
    let mut state_a_serial = model.create_state().unwrap();
    let tokens_a = model.tokenizer().encode(prompt_a).unwrap();
    let mut inp = Tensor::new(&[tokens_a.len()], DataType::I32, DeviceType::Cpu).unwrap();
    inp.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&tokens_a);
    let mut pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu).unwrap();
    pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_a = model.forward_prefill(&mut state_a_serial, &inp, &pos, tokens_a.len()).unwrap();
    let mut serial_a = vec![first_a];
    let mut inp1 = Tensor::new(&[1], DataType::I32, DeviceType::Cpu).unwrap();
    for step in 0..num_decode_steps {
        pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = (tokens_a.len() + step) as i32;
        inp1.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = *serial_a.last().unwrap();
        serial_a.push(model.forward_decoding(&mut state_a_serial, &inp1, &pos).unwrap());
    }

    // ═══ 串行 baseline for prompt B ═══
    let mut state_b_serial = model.create_state().unwrap();
    let tokens_b = model.tokenizer().encode(prompt_b).unwrap();
    let mut inp_b = Tensor::new(&[tokens_b.len()], DataType::I32, DeviceType::Cpu).unwrap();
    inp_b.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&tokens_b);
    pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_b = model.forward_prefill(&mut state_b_serial, &inp_b, &pos, tokens_b.len()).unwrap();
    let mut serial_b = vec![first_b];
    for step in 0..num_decode_steps {
        pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = (tokens_b.len() + step) as i32;
        inp1.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = *serial_b.last().unwrap();
        serial_b.push(model.forward_decoding(&mut state_b_serial, &inp1, &pos).unwrap());
    }

    tracing::info!("Serial A: {:?}", &serial_a);
    tracing::info!("Serial B: {:?}", &serial_b);

    // ═══ Batch decode (2 seqs) ═══
    let mut state_a = model.create_state().unwrap();
    let mut state_b = model.create_state().unwrap();

    // Prefill (串行，因为 seq_len 不同)
    let mut inp_a2 = Tensor::new(&[tokens_a.len()], DataType::I32, DeviceType::Cpu).unwrap();
    inp_a2.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&tokens_a);
    pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_a2 = model.forward_prefill(&mut state_a, &inp_a2, &pos, tokens_a.len()).unwrap();

    let mut inp_b2 = Tensor::new(&[tokens_b.len()], DataType::I32, DeviceType::Cpu).unwrap();
    inp_b2.as_i32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(&tokens_b);
    pos.as_i32_mut().unwrap().as_slice_mut().unwrap()[0] = 0;
    let first_b2 = model.forward_prefill(&mut state_b, &inp_b2, &pos, tokens_b.len()).unwrap();

    assert_eq!(first_a2, first_a);
    assert_eq!(first_b2, first_b);

    let mut workspace = BatchWorkspace::new(model.config(), 64, 4, device).unwrap();
    workspace.sin_cache.copy_from(state_a.workspace.get(&infer_worker::model::BufferType::SinCache).unwrap()).unwrap();
    workspace.cos_cache.copy_from(state_a.workspace.get(&infer_worker::model::BufferType::CosCache).unwrap()).unwrap();

    let mut batch_a = vec![first_a2];
    let mut batch_b = vec![first_b2];
    let mut states = [state_a, state_b];

    for step in 0..num_decode_steps {
        let pos_a = (tokens_a.len() + step) as i32;
        let pos_b = (tokens_b.len() + step) as i32;
        let positions = [pos_a, pos_b];

        let result = model.forward_batch_decode(&mut states, &mut workspace, &positions, None).unwrap();
        assert_eq!(result.len(), 2);
        batch_a.push(result[0]);
        batch_b.push(result[1]);
    }

    tracing::info!("Batch  A: {:?}", &batch_a);
    tracing::info!("Batch  B: {:?}", &batch_b);

    // ═══ 对比 ═══
    for i in 0..serial_a.len() {
        assert_eq!(serial_a[i], batch_a[i], "A mismatch at {}: serial={}, batch={}", i, serial_a[i], batch_a[i]);
    }
    for i in 0..serial_b.len() {
        assert_eq!(serial_b[i], batch_b[i], "B mismatch at {}: serial={}, batch={}", i, serial_b[i], batch_b[i]);
    }
    tracing::info!("All tokens match for both seqs!");
}
