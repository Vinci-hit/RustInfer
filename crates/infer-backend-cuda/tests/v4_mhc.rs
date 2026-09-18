//! Independent FP64 mapping/Sinkhorn oracle and explicit BF16 Post boundaries.
use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::exec::ExecScope;
use infer_core::ports::FusedOps;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

fn scope() -> CudaScope {
    Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope()
}
fn data(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 65536) as f32 / 32768.0 - 1.0
        })
        .collect()
}
fn bf(n: usize, seed: u32) -> Vec<bf16> {
    data(n, seed).into_iter().map(bf16::from_f32).collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

struct Case {
    n: usize,
    d: usize,
    x: Tensor<bf16, Cuda>,
    weight: Tensor<f32, Cuda>,
    scale: Tensor<f32, Cuda>,
    base: Tensor<f32, Cuda>,
    scratch: Tensor<f32, Cuda>,
    collapsed: Tensor<bf16, Cuda>,
    post: Tensor<f32, Cuda>,
    comb: Tensor<f32, Cuda>,
    branch: Tensor<bf16, Cuda>,
    output: Tensor<bf16, Cuda>,
}
impl Case {
    fn new(s: &CudaScope, n: usize, d: usize) -> Self {
        let dev = s.device();
        let weights: Vec<_> = data(24 * 4 * d, 47)
            .into_iter()
            .map(|v| v / (4.0 * d as f32).sqrt())
            .collect();
        Self {
            n,
            d,
            x: Tensor::from_host_slice(&bf(n * 4 * d, 17), [n, 4, d], dev).unwrap(),
            weight: Tensor::from_host_slice(&weights, [24, 4 * d], dev).unwrap(),
            scale: Tensor::from_host_slice(&[0.7f32, -1.3, 2.1], [3], dev).unwrap(),
            base: Tensor::from_host_slice(&data(24, 113), [24], dev).unwrap(),
            scratch: Tensor::from_host_slice(
                &vec![f32::NAN; Cuda::v4_mhc_workspace_floats(n, d, false).unwrap()],
                [Cuda::v4_mhc_workspace_floats(n, d, false).unwrap()],
                dev,
            )
            .unwrap(),
            collapsed: Tensor::from_host_slice(&vec![bf16::NAN; n * d], [n, d], dev).unwrap(),
            post: Tensor::from_host_slice(&vec![f32::NAN; n * 4], [n, 4], dev).unwrap(),
            comb: Tensor::from_host_slice(&vec![f32::NAN; n * 16], [n, 4, 4], dev).unwrap(),
            branch: Tensor::from_host_slice(&bf(n * d, 373), [n, d], dev).unwrap(),
            output: Tensor::from_host_slice(&vec![bf16::NAN; n * 4 * d], [n, 4, d], dev).unwrap(),
        }
    }
    fn pre(&mut self, s: &CudaScope, iters: usize) {
        Cuda::v4_mhc_pre(
            s,
            &self.x,
            &self.weight,
            &self.scale,
            &self.base,
            &mut self.scratch,
            &mut self.collapsed,
            &mut self.post,
            &mut self.comb,
            1e-6,
            1e-6,
            iters,
        )
        .unwrap();
    }
    fn post(&mut self, s: &CudaScope) {
        Cuda::v4_mhc_post(
            s,
            &self.x,
            &self.branch,
            &self.post,
            &self.comb,
            &mut self.output,
        )
        .unwrap();
    }
    fn check(&self, iters: usize) {
        let x = self.x.to_host_vec().unwrap();
        let w = self.weight.to_host_vec().unwrap();
        let scale = self.scale.to_host_vec().unwrap();
        let base = self.base.to_host_vec().unwrap();
        let post = self.post.to_host_vec().unwrap();
        let comb = self.comb.to_host_vec().unwrap();
        let collapsed = self.collapsed.to_host_vec().unwrap();
        for t in 0..self.n {
            let row = &x[t * 4 * self.d..(t + 1) * 4 * self.d];
            let (y, p, c) = oracle(row, &w, &scale, &base, iters);
            close_bf(&collapsed[t * self.d..(t + 1) * self.d], &y);
            close(&post[t * 4..(t + 1) * 4], &p);
            close(&comb[t * 16..(t + 1) * 16], &c);
        }
    }
    fn check_post(&self) {
        let expected = post_reference(
            &self.x.to_host_vec().unwrap(),
            &self.branch.to_host_vec().unwrap(),
            &self.post.to_host_vec().unwrap(),
            &self.comb.to_host_vec().unwrap(),
            self.d,
        );
        assert_eq!(self.output.to_host_vec().unwrap(), expected);
    }
}

// FP64 scalar operations, normalize X *before* the dot as in the CPU reference.
// No split-K, token tiling or CUDA warp reduction is mirrored here.
fn oracle(
    x: &[bf16],
    w: &[f32],
    scale: &[f32],
    base: &[f32],
    iters: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let inv = (x.iter().map(|v| v.to_f64().powi(2)).sum::<f64>() / x.len() as f64 + 1e-6)
        .sqrt()
        .recip();
    let mix: Vec<f64> = w
        .chunks_exact(x.len())
        .map(|w| {
            x.iter()
                .zip(w)
                .map(|(v, w)| v.to_f64() * inv * f64::from(*w))
                .sum()
        })
        .collect();
    let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
    let pre: Vec<_> = (0..4)
        .map(|i| sigmoid(mix[i] * f64::from(scale[0]) + f64::from(base[i])) + 1e-6)
        .collect();
    let d = x.len() / 4;
    let collapsed = (0..d)
        .map(|z| (0..4).map(|i| pre[i] * x[i * d + z].to_f64()).sum())
        .collect();
    if mix.len() == 4 {
        return (collapsed, Vec::new(), Vec::new());
    }
    let post = (0..4)
        .map(|i| 2.0 * sigmoid(mix[4 + i] * f64::from(scale[1]) + f64::from(base[4 + i])))
        .collect();
    let mut comb = vec![0.0; 16];
    for i in 0..4 {
        let logits: Vec<_> = (0..4)
            .map(|j| mix[8 + i * 4 + j] * f64::from(scale[2]) + f64::from(base[8 + i * 4 + j]))
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|v| (v - max).exp()).sum();
        for j in 0..4 {
            comb[i * 4 + j] = (logits[j] - max).exp() / total + 1e-6;
        }
    }
    for iteration in 0..iters {
        if iteration > 0 {
            for row in comb.chunks_exact_mut(4) {
                let total = row.iter().sum::<f64>() + 1e-6;
                for v in row {
                    *v /= total;
                }
            }
        }
        for j in 0..4 {
            let total = (0..4).map(|i| comb[i * 4 + j]).sum::<f64>() + 1e-6;
            for i in 0..4 {
                comb[i * 4 + j] /= total;
            }
        }
    }
    (collapsed, post, comb)
}
fn close(a: &[f32], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && (f64::from(a) - b).abs() < 3e-6,
            "FP32 {i}: {a} vs {b}"
        );
    }
}
fn close_bf(a: &[bf16], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        // Half a BF16 ULP plus FP32 mapping/reduction error. Compare to the
        // unrounded oracle, including values close to a BF16 midpoint.
        let tol = b.abs() / 256.0 + 3e-6;
        assert!(
            a.is_finite() && (a.to_f64() - b).abs() <= tol,
            "BF16 {i}: {a} vs {b}, tol {tol}"
        );
    }
}
fn post_reference(x: &[bf16], branch: &[bf16], post: &[f32], comb: &[f32], d: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; x.len()];
    for t in 0..branch.len() / d {
        for j in 0..4 {
            for z in 0..d {
                let mix: f32 = (0..4)
                    .map(|i| round(comb[t * 16 + i * 4 + j]) * x[(t * 4 + i) * d + z].to_f32())
                    .sum();
                let placed = round(round(post[t * 4 + j]) * branch[t * d + z].to_f32());
                out[(t * 4 + j) * d + z] = bf16::from_f32(round(mix) + placed);
            }
        }
    }
    out
}

#[test]
fn scratch_sizes_reject_invalid_or_overflowing_dimensions() {
    assert_eq!(Cuda::v4_mhc_workspace_floats(1, 4096, false).unwrap(), 1600);
    assert_eq!(Cuda::v4_mhc_workspace_floats(7, 6, true).unwrap(), 35);
    for (n, d) in [
        (0, 128),
        (1, 0),
        (1, 1),
        (1, 129),
        (1, 8194),
        (usize::MAX, 4096),
        (i32::MAX as usize, 4096),
    ] {
        assert!(Cuda::v4_mhc_workspace_floats(n, d, false).is_err());
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn pre_and_head_match_fp64_at_tiny_flash_and_partial_tiles() {
    let s = scope();
    let _active = s.enter();
    for (n, d, iters) in [
        (1, 2, 1),
        (7, 6, 2),
        (1, 32, 20),
        (17, 128, 20),
        (3, 320, 2),
        (5, 4096, 20),
        (1, 8192, 20),
    ] {
        let mut c = Case::new(&s, n, d);
        let head_weight = c.weight.narrow(0, 0, 4).unwrap();
        let head_scale = c.scale.narrow(0, 0, 1).unwrap();
        let head_base = c.base.narrow(0, 0, 4).unwrap();
        let mut head = Tensor::zeros([n, d], s.device()).unwrap();
        let mut scratch = Tensor::zeros(
            [Cuda::v4_mhc_workspace_floats(n, d, true).unwrap()],
            s.device(),
        )
        .unwrap();
        c.pre(&s, iters);
        c.post(&s);
        Cuda::v4_mhc_head(
            &s,
            &c.x,
            &head_weight,
            &head_scale,
            &head_base,
            &mut scratch,
            &mut head,
            1e-6,
            1e-6,
        )
        .unwrap();
        s.synchronize().unwrap();
        c.check(iters);
        c.check_post();
        assert_eq!(
            head.to_host_vec().unwrap(),
            c.collapsed.to_host_vec().unwrap()
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn post_transposes_asymmetric_matrix_and_preserves_bf16_rounding() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 3, 320);
    c.post
        .upload_from_host(
            &data(12, 97)
                .into_iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    c.comb.upload_from_host(&data(48, 77)).unwrap();
    c.post(&s);
    s.synchronize().unwrap();
    c.check_post();
    // A directed permutation is an exact, easily diagnosed transpose oracle.
    c.post.upload_from_host(&[0.0; 12]).unwrap();
    let mut comb = vec![0.0; 48];
    for t in 0..3 {
        for i in 0..4 {
            comb[t * 16 + i * 4 + (i + 1) % 4] = 1.0;
        }
    }
    c.comb.upload_from_host(&comb).unwrap();
    c.post(&s);
    s.synchronize().unwrap();
    let x = c.x.to_host_vec().unwrap();
    let out = c.output.to_host_vec().unwrap();
    for t in 0..3 {
        for j in 0..4 {
            assert_eq!(
                &out[(t * 4 + j) * 320..(t * 4 + j + 1) * 320],
                &x[(t * 4 + (j + 3) % 4) * 320..(t * 4 + (j + 3) % 4 + 1) * 320]
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn zero_input_and_saturated_logits_keep_sinkhorn_finite() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 7, 128);
    c.x.upload_from_host(&vec![bf16::ZERO; 7 * 4 * 128])
        .unwrap();
    let mut base = vec![0.0; 24];
    base[..8].copy_from_slice(&[-100.0, 100.0, 0.0, -20.0, -100.0, 100.0, 0.0, 20.0]);
    for i in 0..4 {
        for j in 0..4 {
            base[8 + i * 4 + j] = if i == j { 100.0 } else { -100.0 };
        }
    }
    c.base.upload_from_host(&base).unwrap();
    for iters in [1, 2, 20] {
        c.pre(&s, iters);
        s.synchronize().unwrap();
        c.check(iters);
        assert!(
            c.collapsed
                .to_host_vec()
                .unwrap()
                .iter()
                .all(|v| *v == bf16::ZERO)
        );
        for matrix in c.comb.to_host_vec().unwrap().chunks_exact(16) {
            for i in 0..4 {
                assert!((matrix[i * 4..i * 4 + 4].iter().sum::<f32>() - 1.0).abs() < 2e-6);
                assert!(((0..4).map(|j| matrix[j * 4 + i]).sum::<f32>() - 1.0).abs() < 2e-6);
            }
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn full_chunks_and_decode_are_bitwise_equal_without_allocating() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 137, 128);
    let collapsed = Tensor::zeros([137, 128], s.device()).unwrap();
    let post = Tensor::zeros([137, 4], s.device()).unwrap();
    let comb = Tensor::zeros([137, 4, 4], s.device()).unwrap();
    c.pre(&s, 20);
    let before = s.device().config.pool_stats();
    for counts in [vec![3, 1, 11, 1, 111, 1, 1, 8], vec![1; 137]] {
        let mut start = 0;
        for n in counts {
            Cuda::v4_mhc_pre(
                &s,
                &c.x.narrow(0, start, n).unwrap(),
                &c.weight,
                &c.scale,
                &c.base,
                &mut c.scratch,
                &mut collapsed.narrow(0, start, n).unwrap(),
                &mut post.narrow(0, start, n).unwrap(),
                &mut comb.narrow(0, start, n).unwrap(),
                1e-6,
                1e-6,
                20,
            )
            .unwrap();
            start += n;
        }
        assert_eq!(start, 137);
        s.synchronize().unwrap();
        assert_eq!(
            c.collapsed.to_host_vec().unwrap(),
            collapsed.to_host_vec().unwrap()
        );
        assert_eq!(c.post.to_host_vec().unwrap(), post.to_host_vec().unwrap());
        assert_eq!(c.comb.to_host_vec().unwrap(), comb.to_host_vec().unwrap());
    }
    assert_eq!(s.device().config.pool_stats(), before);
    c.check(20);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn captured_pre_post_head_replay_reads_new_device_contents() {
    let s = scope();
    let _active = s.enter();
    let mut c = Case::new(&s, 5, 320);
    let weight = c.weight.narrow(0, 0, 4).unwrap();
    let scale = c.scale.narrow(0, 0, 1).unwrap();
    let base = c.base.narrow(0, 0, 4).unwrap();
    let mut head = Tensor::zeros([5, 320], s.device()).unwrap();
    s.synchronize().unwrap();
    s.graph_capture_begin().unwrap();
    c.pre(&s, 20);
    c.post(&s);
    Cuda::v4_mhc_head(
        &s,
        &c.output,
        &weight,
        &scale,
        &base,
        &mut c.scratch,
        &mut head,
        1e-6,
        1e-6,
    )
    .unwrap();
    s.graph_capture_end(4401).unwrap();
    let before = s.device().config.pool_stats();
    for seed in [177, 319, 11] {
        c.x.upload_from_host(&bf(5 * 4 * 320, seed)).unwrap();
        c.weight
            .upload_from_host(
                &data(24 * 4 * 320, seed + 1)
                    .into_iter()
                    .map(|v| v * 0.02)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        c.branch.upload_from_host(&bf(5 * 320, seed + 2)).unwrap();
        c.base.upload_from_host(&data(24, seed + 3)).unwrap();
        c.scale.upload_from_host(&data(3, seed + 4)).unwrap();
        s.graph_launch(4401).unwrap();
        s.synchronize().unwrap();
        c.check(20);
        c.check_post();
        let out = c.output.to_host_vec().unwrap();
        let actual = head.to_host_vec().unwrap();
        let w = weight.to_host_vec().unwrap();
        let sc = scale.to_host_vec().unwrap();
        let b = base.to_host_vec().unwrap();
        for t in 0..5 {
            let (expected, _, _) = oracle(&out[t * 4 * 320..(t + 1) * 4 * 320], &w, &sc, &b, 1);
            close_bf(&actual[t * 320..(t + 1) * 320], &expected);
        }
    }
    assert_eq!(s.device().config.pool_stats(), before);
    s.device().config.invalidate_all_graphs();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn shape_alignment_alias_and_scratch_guards() {
    let s = scope();
    let _active = s.enter();
    let dev = s.device();
    let mut c = Case::new(&s, 3, 6);
    let words = Cuda::v4_mhc_workspace_floats(3, 6, false).unwrap();
    let scratch_guard =
        Tensor::from_host_slice(&vec![-987f32; words + 2], [words + 2], dev).unwrap();
    c.scratch = scratch_guard.narrow(0, 1, words).unwrap();
    let out_guard = Tensor::from_host_slice(&[bf16::from_f32(-987.0); 22], [22], dev).unwrap();
    c.collapsed = out_guard
        .narrow(0, 2, 18)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[3, 6]))
        .unwrap();
    c.pre(&s, 20);
    s.synchronize().unwrap();
    c.check(20);
    let scratch = scratch_guard.to_host_vec().unwrap();
    assert_eq!(scratch[0], -987.0);
    assert_eq!(scratch[words + 1], -987.0);
    let out = out_guard.to_host_vec().unwrap();
    assert_eq!(&out[..2], &[bf16::from_f32(-987.0); 2]);
    assert_eq!(&out[20..], &[bf16::from_f32(-987.0); 2]);
    let mut call = |x: &Tensor<bf16, Cuda>,
                    weight: &Tensor<f32, Cuda>,
                    scratch: &mut Tensor<f32, Cuda>,
                    collapsed: &mut Tensor<bf16, Cuda>,
                    eps,
                    iters| {
        Cuda::v4_mhc_pre(
            &s,
            x,
            weight,
            &c.scale,
            &c.base,
            scratch,
            collapsed,
            &mut c.post,
            &mut c.comb,
            eps,
            1e-6,
            iters,
        )
    };
    let mut short = c.scratch.narrow(0, 0, words - 1).unwrap();
    assert!(call(&c.x, &c.weight, &mut short, &mut c.collapsed, 1e-6, 20).is_err());
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(call(&c.x, &c.weight, &mut c.scratch, &mut c.collapsed, eps, 20).is_err());
    }
    for iters in [0, 21, usize::MAX] {
        assert!(
            call(
                &c.x,
                &c.weight,
                &mut c.scratch,
                &mut c.collapsed,
                1e-6,
                iters
            )
            .is_err()
        );
    }
    let strided = Tensor::<bf16, _>::zeros([3, 4, 8], dev)
        .unwrap()
        .narrow(2, 0, 6)
        .unwrap();
    assert!(
        call(
            &strided,
            &c.weight,
            &mut c.scratch,
            &mut c.collapsed,
            1e-6,
            20
        )
        .is_err()
    );
    let misaligned = Tensor::<bf16, _>::zeros([73], dev)
        .unwrap()
        .narrow(0, 1, 72)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[3, 4, 6]))
        .unwrap();
    assert!(
        call(
            &misaligned,
            &c.weight,
            &mut c.scratch,
            &mut c.collapsed,
            1e-6,
            20
        )
        .is_err()
    );
    let mut alias =
        c.x.clone()
            .view_contiguous(Shape::from_slice(&[72]))
            .unwrap()
            .narrow(0, 0, 18)
            .unwrap()
            .view_contiguous(Shape::from_slice(&[3, 6]))
            .unwrap();
    assert!(call(&c.x, &c.weight, &mut c.scratch, &mut alias, 1e-6, 20).is_err());
    let mut weight_alias = c
        .weight
        .clone()
        .view_contiguous(Shape::from_slice(&[576]))
        .unwrap()
        .narrow(0, 0, words)
        .unwrap();
    assert!(
        call(
            &c.x,
            &c.weight,
            &mut weight_alias,
            &mut c.collapsed,
            1e-6,
            20
        )
        .is_err()
    );
    let wrong = Tensor::zeros([24, 12], dev).unwrap();
    assert!(call(&c.x, &wrong, &mut c.scratch, &mut c.collapsed, 1e-6, 20).is_err());
    let mut residual_alias = c.x.clone();
    assert!(Cuda::v4_mhc_post(&s, &c.x, &c.branch, &c.post, &c.comb, &mut residual_alias).is_err());
    let mut overlapping = c
        .comb
        .clone()
        .view_contiguous(Shape::from_slice(&[48]))
        .unwrap()
        .narrow(0, 0, 12)
        .unwrap()
        .view_contiguous(Shape::from_slice(&[3, 4]))
        .unwrap();
    assert!(
        Cuda::v4_mhc_pre(
            &s,
            &c.x,
            &c.weight,
            &c.scale,
            &c.base,
            &mut c.scratch,
            &mut c.collapsed,
            &mut overlapping,
            &mut c.comb,
            1e-6,
            1e-6,
            20
        )
        .is_err()
    );
    assert!(
        Cuda::v4_mhc_head(
            &s,
            &c.x,
            &c.weight,
            &c.scale,
            &c.base,
            &mut c.scratch,
            &mut c.collapsed,
            1e-6,
            1e-6
        )
        .is_err()
    );
}
