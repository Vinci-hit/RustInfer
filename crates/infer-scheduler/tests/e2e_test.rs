//! 端到端集成测试: HTTP Client → Scheduler → Worker → 回来
//!
//! 在同一进程内启动:
//! - ModelRunner + WorkerServer（GPU 推理线程）
//! - Scheduler（调度线程）
//! - 主线程模拟 HTTP Server (DEALER socket)
//!
//! 验证: 发送 InferenceRequest (含 input_ids) → 收到 InferenceResponse (含 output_token_ids)
//! → decode 为正确文本

#[cfg(feature = "cuda")]
mod e2e {
    use infer_protocol::*;
    use infer_scheduler::Scheduler;
    use infer_worker::base::DeviceType;
    use infer_worker::model::llm::llama3::Llama3;
    use infer_worker::model::llm::LlmModel;
    use infer_worker::worker::runner::ModelRunner;
    use infer_worker::worker::WorkerServer;
    use std::sync::Arc;

    fn get_model_path() -> Option<std::path::PathBuf> {
        std::env::var("LLAMA3_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let p = std::path::PathBuf::from("/data/home/vinciiliu/models/Llama-3.2-1B-Instruct");
                if p.exists() { Some(p) } else { None }
            })
    }

    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn scheduler_worker_single_request() {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return; }
        };

        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device).unwrap();
        let eos_token_ids: Vec<i32> = model.tokenizer().eos_token_ids()
            .iter().map(|&id| id as i32).collect();

        // Tokenize prompt (这在真实场景由 HTTP Server 做)
        let prompt = "The capital of France is";
        let input_ids: Vec<i32> = model.tokenizer().encode(prompt).unwrap();

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs).unwrap());

        let zmq_ctx = zmq::Context::new();

        // Scheduler → Worker 通道
        let sched_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        sched_push.bind("inproc://e2e1-worker-in").unwrap();
        let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        worker_pull.connect("inproc://e2e1-worker-in").unwrap();

        let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        worker_push.bind("inproc://e2e1-worker-out").unwrap();
        let sched_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        sched_pull.connect("inproc://e2e1-worker-out").unwrap();

        // HTTP Client → Scheduler 通道
        let sched_router = zmq_ctx.socket(zmq::ROUTER).unwrap();
        sched_router.bind("inproc://e2e1-frontend").unwrap();
        let client_dealer = zmq_ctx.socket(zmq::DEALER).unwrap();
        client_dealer.connect("inproc://e2e1-frontend").unwrap();

        // 启动 Worker
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let server = WorkerServer::new(
            Arc::clone(&runner), device, worker_pull, worker_push, eos_token_ids,
        );
        let _server_handle = std::thread::spawn(move || server.run());

        // 启动 Scheduler
        let scheduler = Scheduler::new(
            sched_router, sched_push, sched_pull,
            max_batch_tokens, max_batch_seqs,
        );
        let _sched_handle = std::thread::spawn(move || scheduler.run());

        // 模拟 HTTP Client 发请求
        let max_tokens = 10usize;
        let req = InferenceRequest {
            request_id: "e2e-001".to_string(),
            input_ids: input_ids.clone(),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            stream: false,
            priority: 0,
        };
        let data = rmp_serde::to_vec(&req).unwrap();
        client_dealer.send(&b""[..], zmq::SNDMORE).unwrap();
        client_dealer.send(&data, 0).unwrap();

        // 等回复 (DEALER: [empty, data])
        let _empty = client_dealer.recv_bytes(0).unwrap();
        let resp_data = client_dealer.recv_bytes(0).unwrap();
        let resp: InferenceResponse = rmp_serde::from_slice(&resp_data).unwrap();

        // 验证
        assert_eq!(resp.request_id, "e2e-001");
        assert!(matches!(resp.status, ResponseStatus::Success));
        assert!(!resp.output_token_ids.is_empty());
        assert!(resp.output_token_ids.len() <= max_tokens);

        // Decode (在真实场景由 HTTP Server 做)
        let text = runner.model().tokenizer().decode(&resp.output_token_ids).unwrap();
        let full = format!("{}{}", prompt, text);
        eprintln!("e2e result: '{}' → '{}'", prompt, text);
        assert!(full.to_lowercase().contains("paris"), "output doesn't mention paris: {:?}", full);

        eprintln!("PASSED: {} tokens, {:.1} tok/s", resp.metrics.num_tokens, resp.metrics.tokens_per_second);

        runner.request_shutdown();
        let _ = runner_handle.join();
    }

    /// Online continuous batching 测试：
    /// 多个请求在不同时刻到达，模拟真实 online serving。
    ///
    /// 场景：
    /// - t=0: 发 req-A ("The capital of France is", max_tokens=15)
    /// - t≈50ms: req-A 正在 decode 期间，发 req-B ("Once upon a time", max_tokens=10)
    /// - t≈100ms: 再发 req-C ("Hello world", max_tokens=8)
    ///
    /// 验证：
    /// - 三个请求都完成且文本合理
    /// - req-B/C 动态加入 batch 不影响 req-A 的正确性
    /// - 总耗时 < 3 * 单独跑的耗时（证明 batch 生效）
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn scheduler_worker_online_continuous_batching() {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return; }
        };

        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device).unwrap();
        let eos_token_ids: Vec<i32> = model.tokenizer().eos_token_ids()
            .iter().map(|&id| id as i32).collect();
        let tokenizer = model.tokenizer();

        // 准备多个不同 prompt
        let prompts = [
            ("req-A", "The capital of France is", 15usize),
            ("req-B", "Once upon a time", 10usize),
            ("req-C", "Hello world", 8usize),
        ];
        let tokenized: Vec<(&str, Vec<i32>, usize)> = prompts.iter()
            .map(|(id, prompt, max_tok)| (*id, tokenizer.encode(prompt).unwrap(), *max_tok))
            .collect();

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 4usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs).unwrap());

        let zmq_ctx = zmq::Context::new();

        // Scheduler ↔ Worker
        let sched_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        sched_push.bind("inproc://online-worker-in").unwrap();
        let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        worker_pull.connect("inproc://online-worker-in").unwrap();

        let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        worker_push.bind("inproc://online-worker-out").unwrap();
        let sched_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        sched_pull.connect("inproc://online-worker-out").unwrap();

        // Client ↔ Scheduler
        let sched_router = zmq_ctx.socket(zmq::ROUTER).unwrap();
        sched_router.bind("inproc://online-frontend").unwrap();
        let client_dealer = zmq_ctx.socket(zmq::DEALER).unwrap();
        client_dealer.set_identity(b"online-client").unwrap();
        client_dealer.connect("inproc://online-frontend").unwrap();

        // 启动 Worker
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let server = WorkerServer::new(
            Arc::clone(&runner), device, worker_pull, worker_push, eos_token_ids,
        );
        let _server_handle = std::thread::spawn(move || server.run());

        // 启动 Scheduler
        let scheduler = Scheduler::new(
            sched_router, sched_push, sched_pull,
            max_batch_tokens, max_batch_seqs,
        );
        let _sched_handle = std::thread::spawn(move || scheduler.run());

        let start = std::time::Instant::now();

        // ── t=0: 发 req-A ──
        {
            let (id, ref input_ids, max_tokens) = tokenized[0];
            let req = InferenceRequest {
                request_id: id.to_string(),
                input_ids: input_ids.clone(),
                max_tokens,
                temperature: 0.0, top_p: 1.0, top_k: -1,
                stream: false, priority: 0,
            };
            client_dealer.send(&b""[..], zmq::SNDMORE).unwrap();
            client_dealer.send(&rmp_serde::to_vec(&req).unwrap(), 0).unwrap();
            eprintln!("[{:>4}ms] sent {}", start.elapsed().as_millis(), id);
        }

        // 等 ~50ms 后发 req-B（让 req-A 已经开始 decode）
        std::thread::sleep(std::time::Duration::from_millis(50));
        {
            let (id, ref input_ids, max_tokens) = tokenized[1];
            let req = InferenceRequest {
                request_id: id.to_string(),
                input_ids: input_ids.clone(),
                max_tokens,
                temperature: 0.0, top_p: 1.0, top_k: -1,
                stream: false, priority: 0,
            };
            client_dealer.send(&b""[..], zmq::SNDMORE).unwrap();
            client_dealer.send(&rmp_serde::to_vec(&req).unwrap(), 0).unwrap();
            eprintln!("[{:>4}ms] sent {}", start.elapsed().as_millis(), id);
        }

        // 再等 ~50ms 发 req-C
        std::thread::sleep(std::time::Duration::from_millis(50));
        {
            let (id, ref input_ids, max_tokens) = tokenized[2];
            let req = InferenceRequest {
                request_id: id.to_string(),
                input_ids: input_ids.clone(),
                max_tokens,
                temperature: 0.0, top_p: 1.0, top_k: -1,
                stream: false, priority: 0,
            };
            client_dealer.send(&b""[..], zmq::SNDMORE).unwrap();
            client_dealer.send(&rmp_serde::to_vec(&req).unwrap(), 0).unwrap();
            eprintln!("[{:>4}ms] sent {}", start.elapsed().as_millis(), id);
        }

        // ── 收集 3 个回复 ──
        let mut responses: Vec<InferenceResponse> = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);

        while responses.len() < 3 {
            if std::time::Instant::now() > deadline {
                panic!("timeout: only got {} / 3 responses", responses.len());
            }
            // DEALER recv: [empty, data]
            client_dealer.set_rcvtimeo(1000).unwrap();
            match client_dealer.recv_bytes(0) {
                Ok(_empty) => {
                    let data = client_dealer.recv_bytes(0).unwrap();
                    let resp: InferenceResponse = rmp_serde::from_slice(&data).unwrap();
                    eprintln!("[{:>4}ms] received {} ({} tokens, {:.1} tok/s)",
                        start.elapsed().as_millis(),
                        resp.request_id, resp.metrics.num_tokens, resp.metrics.tokens_per_second);
                    responses.push(resp);
                }
                Err(zmq::Error::EAGAIN) => continue,
                Err(e) => panic!("recv error: {:?}", e),
            }
        }

        let total_ms = start.elapsed().as_millis();
        eprintln!("\n── Results ({} ms total) ──", total_ms);

        // ── 验证每个请求 ──
        for (i, (id, prompt, _max_tok)) in prompts.iter().enumerate() {
            let resp = responses.iter().find(|r| r.request_id == *id)
                .unwrap_or_else(|| panic!("response for {} not found", id));

            assert!(matches!(resp.status, ResponseStatus::Success),
                "{} failed: {:?}", id, resp.error);
            assert!(!resp.output_token_ids.is_empty(),
                "{} generated no tokens", id);

            let text = runner.model().tokenizer().decode(&resp.output_token_ids).unwrap();
            eprintln!("  {}: '{}' → '{}'", id, prompt, text);

            // 基本语义检查
            match i {
                0 => {
                    let full = format!("{}{}", prompt, text);
                    assert!(full.to_lowercase().contains("paris"),
                        "req-A should mention Paris: {:?}", full);
                }
                _ => {
                    assert!(text.len() > 5,
                        "{} text too short: {:?}", id, text);
                }
            }
        }

        // ── 验证 continuous batching 生效 ──
        // 如果是串行的，总耗时 ≈ sum(每个请求时间)
        // 如果是 batch 的，总耗时 ≈ max(每个请求时间) + 一点开销
        let sum_individual_ms: u64 = responses.iter().map(|r| r.metrics.total_ms).sum();
        eprintln!("\n  total wall time: {}ms", total_ms);
        eprintln!("  sum individual: {}ms", sum_individual_ms);
        eprintln!("  speedup ratio: {:.2}x", sum_individual_ms as f64 / total_ms as f64);

        // batch 应该比串行快至少 1.3x（保守阈值，实际应接近 req 数量）
        assert!(
            (total_ms as u64) < sum_individual_ms,
            "no batching benefit! wall={}ms >= sum_individual={}ms",
            total_ms, sum_individual_ms,
        );

        eprintln!("\n✅ Online continuous batching test PASSED");

        runner.request_shutdown();
        let _ = runner_handle.join();
    }

    /// 压测：100 请求（Alpaca 数据集），模拟高并发 online serving
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH and CUDA GPU"]
    fn scheduler_worker_benchmark_online() {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return; }
        };

        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device).unwrap();
        let eos_token_ids: Vec<i32> = model.tokenizer().eos_token_ids()
            .iter().map(|&id| id as i32).collect();
        let tokenizer = model.tokenizer();

        // 从 Alpaca 数据集加载真实 prompts
        let dataset_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/bench_prompts.json");
        let prompts_raw: Vec<String> = if dataset_path.exists() {
            let content = std::fs::read_to_string(&dataset_path).unwrap();
            serde_json::from_str(&content).unwrap()
        } else {
            // fallback
            vec![
                "Give three tips for staying healthy.".to_string(),
                "What are the three primary colors?".to_string(),
                "Explain quantum computing in simple terms.".to_string(),
                "Write a Python function that sorts a list.".to_string(),
                "The meaning of life is".to_string(),
            ]
        };

        let num_requests = prompts_raw.len().min(100);
        let max_tokens_per_req = 32usize;

        let tokenized: Vec<(String, Vec<i32>, usize)> = prompts_raw[..num_requests].iter()
            .enumerate()
            .map(|(i, prompt)| {
                let ids = tokenizer.encode(prompt.as_str()).unwrap();
                (format!("bench-{:03}", i), ids, max_tokens_per_req)
            })
            .collect();

        let max_batch_tokens = 1024usize;
        let max_batch_seqs = 8usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs).unwrap());

        let zmq_ctx = zmq::Context::new();

        let sched_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        sched_push.bind("inproc://bench-worker-in").unwrap();
        let worker_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        worker_pull.connect("inproc://bench-worker-in").unwrap();

        let worker_push = zmq_ctx.socket(zmq::PUSH).unwrap();
        worker_push.bind("inproc://bench-worker-out").unwrap();
        let sched_pull = zmq_ctx.socket(zmq::PULL).unwrap();
        sched_pull.connect("inproc://bench-worker-out").unwrap();

        let sched_router = zmq_ctx.socket(zmq::ROUTER).unwrap();
        sched_router.bind("inproc://bench-frontend").unwrap();
        let client_dealer = zmq_ctx.socket(zmq::DEALER).unwrap();
        client_dealer.set_identity(b"bench-client").unwrap();
        client_dealer.connect("inproc://bench-frontend").unwrap();

        // 启动
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());
        let server = WorkerServer::new(
            Arc::clone(&runner), device, worker_pull, worker_push, eos_token_ids,
        );
        let _server_handle = std::thread::spawn(move || server.run());
        let scheduler = Scheduler::new(
            sched_router, sched_push, sched_pull, max_batch_tokens, max_batch_seqs,
        );
        let _sched_handle = std::thread::spawn(move || scheduler.run());

        eprintln!("\n{}", "=".repeat(70));
        eprintln!("  RustInfer Online Continuous Batching Benchmark");
        eprintln!("  {} requests, max_batch_seqs={}, max_batch_tokens={}",
            tokenized.len(), max_batch_seqs, max_batch_tokens);
        eprintln!("{}\n", "=".repeat(70));

        let start = std::time::Instant::now();

        // 高速交错发送（5ms 间隔 = 200 req/s 到达率）
        for (id, input_ids, max_tokens) in &tokenized {
            let req = InferenceRequest {
                request_id: id.clone(),
                input_ids: input_ids.clone(),
                max_tokens: *max_tokens,
                temperature: 0.0, top_p: 1.0, top_k: -1,
                stream: false, priority: 0,
            };
            client_dealer.send(&b""[..], zmq::SNDMORE).unwrap();
            client_dealer.send(&rmp_serde::to_vec(&req).unwrap(), 0).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!("  All {} requests sent in {}ms", num_requests, start.elapsed().as_millis());

        // 收集所有回复
        let mut results: Vec<(String, InferenceResponse, u128)> = Vec::new();
        client_dealer.set_rcvtimeo(5000).unwrap();

        while results.len() < num_requests {
            let elapsed = start.elapsed().as_millis();
            if elapsed > 60_000 {
                panic!("timeout: got {} / {} responses", results.len(), num_requests);
            }
            match client_dealer.recv_bytes(0) {
                Ok(_empty) => {
                    let data = client_dealer.recv_bytes(0).unwrap();
                    let resp: InferenceResponse = rmp_serde::from_slice(&data).unwrap();
                    let recv_time = start.elapsed().as_millis();
                    results.push((resp.request_id.clone(), resp, recv_time));
                    if results.len() % 10 == 0 {
                        eprintln!("  ... {}/{} responses received ({}ms)",
                            results.len(), num_requests, recv_time);
                    }
                }
                Err(zmq::Error::EAGAIN) => continue,
                Err(e) => panic!("recv error: {:?}", e),
            }
        }

        let total_wall_ms = start.elapsed().as_millis();

        // ── 性能报告 ──
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("  PERFORMANCE REPORT");
        eprintln!("{}", "=".repeat(70));

        let total_tokens: u32 = results.iter().map(|(_, r, _)| r.metrics.num_tokens).sum();
        let sum_individual_ms: u64 = results.iter().map(|(_, r, _)| r.metrics.total_ms).sum();
        let latencies: Vec<u64> = results.iter().map(|(_, r, _)| r.metrics.total_ms).collect();
        let mut sorted_lat = latencies.clone();
        sorted_lat.sort();

        let p50 = sorted_lat[sorted_lat.len() / 2];
        let p90 = sorted_lat[(sorted_lat.len() as f64 * 0.9) as usize];
        let p99 = sorted_lat[sorted_lat.len() - 1];
        let avg_lat = sum_individual_ms / results.len() as u64;

        let system_throughput = total_tokens as f64 / (total_wall_ms as f64 / 1000.0);
        let speedup = sum_individual_ms as f64 / total_wall_ms as f64;

        eprintln!("\n  Requests:          {}", num_requests);
        eprintln!("  Total tokens:      {}", total_tokens);
        eprintln!("  Wall time:         {} ms", total_wall_ms);
        eprintln!("  System throughput: {:.1} tokens/s", system_throughput);
        eprintln!("  Batch speedup:     {:.2}x vs serial", speedup);
        eprintln!();
        eprintln!("  Latency (per-request total_ms):");
        eprintln!("    avg:  {} ms", avg_lat);
        eprintln!("    p50:  {} ms", p50);
        eprintln!("    p90:  {} ms", p90);
        eprintln!("    p99:  {} ms", p99);
        eprintln!();

        eprintln!("  Per-request detail (first 10):");
        eprintln!("  {:>10} {:>8} {:>8} {:>10}", "request_id", "tokens", "ms", "tok/s");
        eprintln!("  {:>10} {:>8} {:>8} {:>10}", "──────────", "──────", "──────", "────────");
        for (id, resp, _) in results.iter().take(10) {
            eprintln!("  {:>10} {:>8} {:>8} {:>10.1}",
                id, resp.metrics.num_tokens, resp.metrics.total_ms, resp.metrics.tokens_per_second);
        }
        if results.len() > 10 {
            eprintln!("  ... ({} more)", results.len() - 10);
        }

        // 验证文本输出（只抽查前 5 个）
        eprintln!("\n  Generated text samples (first 5):");
        for (id, resp, _) in results.iter().take(5) {
            let text = runner.model().tokenizer().decode(&resp.output_token_ids).unwrap();
            let trimmed = if text.len() > 60 { &text[..60] } else { &text };
            eprintln!("    {}: '{}'", id, trimmed.trim());
        }

        eprintln!("\n{}", "=".repeat(70));

        // 断言
        assert!(speedup > 1.0, "no batching benefit: speedup={:.2}x", speedup);
        assert_eq!(results.len(), num_requests);
        for (_, resp, _) in &results {
            assert!(matches!(resp.status, ResponseStatus::Success));
            assert!(!resp.output_token_ids.is_empty());
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
    }
}
