//! CUDA-event graph timings for complete mHC, including FP32 mapping weights.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;

fn measure(s: &CudaScope, key: u64, calls: usize) -> Result<f32, Box<dyn std::error::Error>> {
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
        samples.push(timer.elapsed_ms()?.ok_or("timer incomplete")? * 1000.0 / (5 * calls) as f32);
    }
    assert_eq!(s.device().config.pool_stats(), before);
    samples.sort_by(f32::total_cmp);
    Ok(samples[3])
}
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed) % 1021) as f32 - 510.0) / 511.0)
        .collect()
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
    for (n, d) in [
        (1, 128),
        (128, 128),
        (1, 4096),
        (4, 4096),
        (128, 4096),
        (1024, 4096),
    ] {
        let x = Tensor::from_host_slice(
            &values(n * 4 * d, 7)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, 4, d],
            dev,
        )?;
        let weight = Tensor::from_host_slice(
            &values(24 * 4 * d, 17)
                .into_iter()
                .map(|v| v / (4.0 * d as f32).sqrt())
                .collect::<Vec<_>>(),
            [24, 4 * d],
            dev,
        )?;
        let scale = Tensor::from_host_slice(&[0.7f32, -1.3, 2.1], [3], dev)?;
        let base = Tensor::from_host_slice(&values(24, 19), [24], dev)?;
        let branch = Tensor::from_host_slice(
            &values(n * d, 77)
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            [n, d],
            dev,
        )?;
        let words = Cuda::v4_mhc_workspace_floats(n, d, false)?;
        let mut scratch = Tensor::zeros([words], dev)?;
        let mut collapsed = Tensor::zeros([n, d], dev)?;
        let mut post = Tensor::zeros([n, 4], dev)?;
        let mut comb = Tensor::zeros([n, 4, 4], dev)?;
        let mut output = Tensor::zeros([n, 4, d], dev)?;
        let mut head = Tensor::zeros([n, d], dev)?;
        let head_weight = weight.narrow(0, 0, 4)?;
        let head_scale = scale.narrow(0, 0, 1)?;
        let head_base = base.narrow(0, 0, 4)?;
        Cuda::v4_mhc_pre(
            &s,
            &x,
            &weight,
            &scale,
            &base,
            &mut scratch,
            &mut collapsed,
            &mut post,
            &mut comb,
            1e-6,
            1e-6,
            20,
        )?;
        Cuda::v4_mhc_post(&s, &x, &branch, &post, &comb, &mut output)?;
        s.synchronize()?;
        let calls = if n < 128 { 32 } else { 8 };
        for (key, stage) in [(4501, 0), (4502, 1), (4503, 2), (4504, 3)] {
            s.graph_capture_begin()?;
            for _ in 0..calls {
                if stage == 0 || stage == 3 {
                    Cuda::v4_mhc_pre(
                        &s,
                        &x,
                        &weight,
                        &scale,
                        &base,
                        &mut scratch,
                        &mut collapsed,
                        &mut post,
                        &mut comb,
                        1e-6,
                        1e-6,
                        20,
                    )?;
                }
                if stage == 1 || stage == 3 {
                    Cuda::v4_mhc_post(&s, &x, &branch, &post, &comb, &mut output)?;
                }
                if stage == 2 || stage == 3 {
                    Cuda::v4_mhc_head(
                        &s,
                        &output,
                        &head_weight,
                        &head_scale,
                        &head_base,
                        &mut scratch,
                        &mut head,
                        1e-6,
                        1e-6,
                    )?;
                }
            }
            s.graph_capture_end(key)?;
        }
        let serial_collapsed = Tensor::zeros([n, d], dev)?;
        let serial_post = Tensor::zeros([n, 4], dev)?;
        let serial_comb = Tensor::zeros([n, 4, 4], dev)?;
        s.graph_capture_begin()?;
        for t in 0..n {
            Cuda::v4_mhc_pre(
                &s,
                &x.narrow(0, t, 1)?,
                &weight,
                &scale,
                &base,
                &mut scratch,
                &mut serial_collapsed.narrow(0, t, 1)?,
                &mut serial_post.narrow(0, t, 1)?,
                &mut serial_comb.narrow(0, t, 1)?,
                1e-6,
                1e-6,
                20,
            )?;
        }
        s.graph_capture_end(4505)?;
        let pre_us = measure(&s, 4501, calls)?;
        let post_us = measure(&s, 4502, calls)?;
        let head_us = measure(&s, 4503, calls)?;
        let pipeline_us = measure(&s, 4504, calls)?;
        let serial_pre_us = measure(&s, 4505, 1)?;
        assert_eq!(collapsed.to_host_vec()?, serial_collapsed.to_host_vec()?);
        assert_eq!(post.to_host_vec()?, serial_post.to_host_vec()?);
        assert_eq!(comb.to_host_vec()?, serial_comb.to_host_vec()?);
        assert!(head.to_host_vec()?.iter().all(|v| v.is_finite()));
        println!(
            "{{\"tokens\":{n},\"dim\":{d},\"pre_us\":{pre_us:.3},\"post_us\":{post_us:.3},\"head_us\":{head_us:.3},\"pipeline_us\":{pipeline_us:.3},\"serial_pre_us\":{serial_pre_us:.3},\"scratch_bytes\":{},\"batch_serial_equal\":true}}",
            words * 4
        );
        dev.config.invalidate_all_graphs();
    }
    Ok(())
}
