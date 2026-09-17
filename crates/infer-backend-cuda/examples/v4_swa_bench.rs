//! Full-window kernel microbenchmark, not end-to-end model throughput.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scope = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )?
    .scope();
    let _active = scope.enter();
    let dev = scope.device();
    let values = |n: usize| {
        (0..n)
            .map(|i| bf16::from_f32((i as f32 * 0.137).sin()))
            .collect::<Vec<_>>()
    };
    let q = Tensor::from_host_slice(&values(64 * 512), [64, 512], dev)?;
    let kv = Tensor::from_host_slice(&values(512), [512], dev)?;
    let sink = Tensor::from_host_slice(&[0.0f32; 64], [64], dev)?;
    let pos = Tensor::from_host_slice(&[4096i32], [1], dev)?;
    let mut cache = Tensor::from_host_slice(&values(128 * 512), [128, 512], dev)?;
    let mut out = Tensor::zeros([64, 512], dev)?;
    for _ in 0..100 {
        Cuda::v4_swa_decode(&scope, &q, &kv, &sink, &pos, &mut cache, &mut out)?;
    }
    scope.synchronize()?;
    // Identical full-window workloads; capture a bundle to amortize graph
    // launch gaps when measuring device kernel time with CUDA events.
    for (key, nodes) in [(1, 1), (32, 32)] {
        scope.graph_capture_begin()?;
        for _ in 0..nodes {
            Cuda::v4_swa_decode(&scope, &q, &kv, &sink, &pos, &mut cache, &mut out)?;
        }
        scope.graph_capture_end(key)?;
        scope.graph_launch(key)?;
    }
    scope.synchronize()?;
    let mut timer = scope.create_timer()?.ok_or("CUDA timer unavailable")?;
    let before = dev.config.pool_stats();
    for (label, key, nodes, launches) in [
        ("eager_stream", 0, 1, 1000),
        ("single_node_graph_stream", 1, 1, 1000),
        ("bundled_graph_device", 32, 32, 100),
    ] {
        let mut samples = Vec::new();
        for _ in 0..7 {
            timer.start()?;
            for _ in 0..launches {
                if key == 0 {
                    Cuda::v4_swa_decode(&scope, &q, &kv, &sink, &pos, &mut cache, &mut out)?;
                } else {
                    scope.graph_launch(key)?;
                }
            }
            timer.stop()?;
            scope.synchronize()?;
            samples.push(
                timer.elapsed_ms()?.ok_or("timer incomplete")? * 1000.0 / (launches * nodes) as f32,
            );
        }
        samples.sort_by(f32::total_cmp);
        println!(
            "{{\"mode\":\"{label}\",\"heads\":64,\"window\":128,\"head_dim\":512,\"median_us\":{:.3},\"min_us\":{:.3}}}",
            samples[3], samples[0]
        );
    }
    assert_eq!(
        dev.config.pool_stats(),
        before,
        "operator/replay allocated GPU storage"
    );
    assert!(out.to_host_vec()?.iter().all(|v| v.is_finite()));
    Ok(())
}
