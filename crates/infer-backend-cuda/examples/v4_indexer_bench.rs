//! CUDA-event graph timings for prepared index Q/K and scaled head weights.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;

fn measure(
    s: &CudaScope,
    key: u64,
    calls_per_graph: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut timer = s.create_timer()?.ok_or("CUDA timer unavailable")?;
    for _ in 0..5 {
        s.graph_launch(key)?;
    }
    s.synchronize()?;
    let before = s.device().config.pool_stats();
    let mut samples = Vec::new();
    for _ in 0..7 {
        timer.start()?;
        for _ in 0..5 {
            s.graph_launch(key)?;
        }
        timer.stop()?;
        s.synchronize()?;
        samples.push(
            timer.elapsed_ms()?.ok_or("timer incomplete")? * 1000.0 / (5 * calls_per_graph) as f32,
        );
    }
    assert_eq!(s.device().config.pool_stats(), before);
    samples.sort_by(f32::total_cmp);
    Ok(samples[3])
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 10007) as f32 * 0.137).sin()).collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let s = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )?
    .scope();
    let _active = s.enter();
    let dev = s.device();
    let h = 64;
    let d = 128;
    for (n, start) in [(128, 0), (512, 0), (1024, 0), (128, 32768)] {
        let c = (start + n) / 4;
        let q = Tensor::from_host_slice(
            &values(n * h * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, h, d],
            dev,
        )?;
        let k = Tensor::from_host_slice(
            &values(c * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [c, d],
            dev,
        )?;
        let weights = Tensor::from_host_slice(
            &values(n * h)
                .into_iter()
                .map(|v| v / ((h * d) as f32).sqrt())
                .collect::<Vec<_>>(),
            [n, h],
            dev,
        )?;
        let positions = Tensor::from_host_slice(
            &(start as i32..(start + n) as i32).collect::<Vec<_>>(),
            [n],
            dev,
        )?;
        let pos = positions.narrow(0, 0, 1)?;
        let mut output = Tensor::<f32, _>::zeros([n, c], dev)?;
        let serial = Tensor::<f32, _>::zeros([n, c], dev)?;
        s.synchronize()?;
        s.graph_capture_begin()?;
        for _ in 0..8 {
            Cuda::v4_indexer_scores(&s, &q, &k, &weights, &pos, &mut output)?;
        }
        s.graph_capture_end(40)?;
        s.graph_capture_begin()?;
        for t in 0..n {
            Cuda::v4_indexer_scores(
                &s,
                &q.narrow(0, t, 1)?,
                &k,
                &weights.narrow(0, t, 1)?,
                &positions.narrow(0, t, 1)?,
                &mut serial.narrow(0, t, 1)?,
            )?;
        }
        s.graph_capture_end(41)?;
        let prefill = measure(&s, 40, 8)?;
        let sequential = measure(&s, 41, 1)?;
        let out = output.to_host_vec()?;
        assert_eq!(out, serial.to_host_vec()?);
        for t in 0..n {
            let visible = (start + t + 1) / 4;
            assert!(out[t * c..t * c + visible].iter().all(|v| v.is_finite()));
            assert!(
                out[t * c + visible..(t + 1) * c]
                    .iter()
                    .all(|v| *v == f32::NEG_INFINITY)
            );
        }
        println!(
            "{{\"kind\":\"indexer_prefill\",\"tokens\":{n},\"start\":{start},\"capacity\":{c},\"heads\":{h},\"prefill_us\":{prefill:.3},\"serial_decode_us\":{sequential:.3},\"speedup\":{:.2},\"output_bytes\":{},\"avoided_per_head_scores_bytes\":{},\"bitwise_equal\":true}}",
            sequential / prefill,
            n * c * 4,
            n * h * c * 4
        );
        dev.config.invalidate_all_graphs();
    }
    for c in [32, 256, 8192, 32768, 262144] {
        let q = Tensor::from_host_slice(
            &values(h * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [1, h, d],
            dev,
        )?;
        let k = Tensor::from_host_slice(
            &values(c * d)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [c, d],
            dev,
        )?;
        let weights = Tensor::from_host_slice(
            &values(h)
                .into_iter()
                .map(|v| v / ((h * d) as f32).sqrt())
                .collect::<Vec<_>>(),
            [1, h],
            dev,
        )?;
        let pos = Tensor::from_host_slice(&[(c * 4 - 1) as i32], [1], dev)?;
        let mut output = Tensor::<f32, _>::zeros([1, c], dev)?;
        s.synchronize()?;
        s.graph_capture_begin()?;
        for _ in 0..32 {
            Cuda::v4_indexer_scores(&s, &q, &k, &weights, &pos, &mut output)?;
        }
        s.graph_capture_end(42)?;
        let latency = measure(&s, 42, 32)?;
        assert!(output.to_host_vec()?.iter().all(|v| v.is_finite()));
        println!(
            "{{\"kind\":\"indexer_decode\",\"visible_tokens\":{},\"compressed_keys\":{c},\"heads\":{h},\"latency_us\":{latency:.3}}}",
            c * 4
        );
        dev.config.invalidate_all_graphs();
    }
    Ok(())
}
