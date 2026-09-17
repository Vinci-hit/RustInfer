//! Compare Tensor Core prefill against serial decode using identical tensors.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

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
    let mut timer = scope.create_timer()?.ok_or("CUDA timer unavailable")?;
    for n in [128, 512, 1024] {
        let values = |len: usize| {
            (0..len)
                .map(|i| bf16::from_f32((i as f32 * 0.137).sin()))
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_host_slice(&values(n * 64 * 512), [n, 64, 512], dev)?;
        let kv = Tensor::from_host_slice(&values(n * 512), [n, 512], dev)?;
        let sink = Tensor::from_host_slice(&[0.0f32; 64], [64], dev)?;
        let positions = Tensor::from_host_slice(&(0..n as i32).collect::<Vec<_>>(), [n], dev)?;
        let start = positions.narrow(0, 0, 1)?;
        let mut cache = Tensor::zeros([128, 512], dev)?;
        let mut output = Tensor::zeros([n, 64, 512], dev)?;
        let mut serial_cache = Tensor::zeros([128, 512], dev)?;
        let serial_output = Tensor::zeros([n, 64, 512], dev)?;
        let prefill_key = n as u64 * 2;
        let serial_key = prefill_key + 1;
        scope.synchronize()?;
        scope.graph_capture_begin()?;
        Cuda::v4_swa_prefill(&scope, &q, &kv, &sink, &start, &mut cache, &mut output)?;
        scope.graph_capture_end(prefill_key)?;
        scope.graph_capture_begin()?;
        for t in 0..n {
            let qi = q
                .narrow(0, t, 1)?
                .view_contiguous(Shape::from_slice(&[64, 512]))?;
            let ki = kv
                .narrow(0, t, 1)?
                .view_contiguous(Shape::from_slice(&[512]))?;
            let pi = positions.narrow(0, t, 1)?;
            let mut oi = serial_output
                .narrow(0, t, 1)?
                .view_contiguous(Shape::from_slice(&[64, 512]))?;
            Cuda::v4_swa_decode(&scope, &qi, &ki, &sink, &pi, &mut serial_cache, &mut oi)?;
        }
        scope.graph_capture_end(serial_key)?;
        for _ in 0..5 {
            scope.graph_launch(prefill_key)?;
            scope.graph_launch(serial_key)?;
        }
        scope.synchronize()?;
        let before = dev.config.pool_stats();
        let mut medians = Vec::new();
        for key in [prefill_key, serial_key] {
            let mut samples = Vec::new();
            for _ in 0..7 {
                timer.start()?;
                for _ in 0..5 {
                    scope.graph_launch(key)?;
                }
                timer.stop()?;
                scope.synchronize()?;
                samples.push(timer.elapsed_ms()?.ok_or("timer incomplete")? * 1000.0 / 5.0);
            }
            samples.sort_by(f32::total_cmp);
            medians.push(samples[3]);
        }
        assert_eq!(dev.config.pool_stats(), before);
        // Compare every output, including early causal positions, outside timing.
        let a = output.to_host_vec()?;
        let b = serial_output.to_host_vec()?;
        let mut error = 0.0f32;
        for (a, b) in a.iter().zip(&b) {
            assert!(a.is_finite() && b.is_finite());
            error = error.max((a.to_f32() - b.to_f32()).abs());
        }
        assert!(error <= 0.008, "prefill/decode max error {error}");
        println!(
            "{{\"tokens\":{n},\"heads\":64,\"prefill_us\":{:.3},\"serial_decode_graph_us\":{:.3},\"speedup\":{:.2},\"max_abs_difference\":{error}}}",
            medians[0],
            medians[1],
            medians[1] / medians[0]
        );
        // Destroy graphs before their captured buffers go out of scope.
        dev.config.invalidate_all_graphs();
    }
    Ok(())
}
