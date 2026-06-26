//! Tests relocated out of `domain/tensor.rs` so that `tensor.rs` itself is
//! backend-agnostic and can collapse into `infer-core`. These exercise `Tensor`
//! against concrete backends (`Cpu`/`Cuda`), which live in the worker, through
//! the public Tensor API only.
#[cfg(test)]
mod helper_tests {
    use crate::domain::tensor::Tensor;
    use crate::domain::ports::OpError;
    use crate::infrastructure::cpu::Cpu;
    use half::bf16;

    #[test]
    fn randn_f32_cpu_seeded_is_deterministic() {
        let dev = Cpu;
        let a: Tensor<f32, Cpu> = Tensor::randn([4, 8], &dev, Some(42)).unwrap();
        let b: Tensor<f32, Cpu> = Tensor::randn([4, 8], &dev, Some(42)).unwrap();
        let av = a.to_host_vec().unwrap();
        let bv = b.to_host_vec().unwrap();
        assert_eq!(av, bv, "same seed must produce identical samples");
        assert_eq!(av.len(), 32);
    }

    #[test]
    fn randn_f32_cpu_distribution_within_range() {
        let dev = Cpu;
        let n = 4096usize;
        let t: Tensor<f32, Cpu> = Tensor::randn([n], &dev, Some(7)).unwrap();
        let v = t.to_host_vec().unwrap();
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        // Standard normal: mean ≈ 0, var ≈ 1. 4k samples → ±0.1 tolerance.
        assert!(mean.abs() < 0.1, "mean was {}", mean);
        assert!((var - 1.0).abs() < 0.15, "var was {}", var);
        // Should not produce all zeros / all NaN.
        assert!(v.iter().any(|&x| x.abs() > 0.1));
        assert!(v.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn randn_bf16_cpu_distribution_within_range() {
        let dev = Cpu;
        let n = 4096usize;
        let t: Tensor<bf16, Cpu> = Tensor::randn([n], &dev, Some(11)).unwrap();
        let v: Vec<f32> = t
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.1, "mean was {}", mean);
        assert!((var - 1.0).abs() < 0.20, "var was {}", var); // wider tol for bf16 quantization
        assert!(v.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn copy_from_cpu_roundtrip() {
        let dev = Cpu;
        let src: Tensor<f32, Cpu> = Tensor::randn([3, 5], &dev, Some(1)).unwrap();
        let mut dst: Tensor<f32, Cpu> = Tensor::zeros([3, 5], &dev).unwrap();
        dst.copy_from(&src).unwrap();
        assert_eq!(src.to_host_vec().unwrap(), dst.to_host_vec().unwrap());
    }

    #[test]
    fn copy_from_shape_mismatch_errors() {
        let dev = Cpu;
        let src: Tensor<f32, Cpu> = Tensor::zeros([3, 5], &dev).unwrap();
        let mut dst: Tensor<f32, Cpu> = Tensor::zeros([4, 5], &dev).unwrap();
        let err = dst.copy_from(&src).unwrap_err();
        match err {
            OpError::Shape(_) => {}
            other => panic!("expected Shape error, got {:?}", other),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn randn_bf16_cuda_seeded_matches_cpu() {
        use crate::infrastructure::cuda::Cuda;
        let cpu = Cpu;
        let cuda = Cuda::new(0).expect("cuda init");
        let n = 1024usize;
        let cpu_t: Tensor<bf16, Cpu> = Tensor::randn([n], &cpu, Some(123)).unwrap();
        let gpu_t: Tensor<bf16, Cuda> = Tensor::randn([n], &cuda, Some(123)).unwrap();
        // Pull GPU back to host.
        let gpu_host = gpu_t.to_host_vec().unwrap();
        let cpu_host = cpu_t.to_host_vec().unwrap();
        for (i, (a, b)) in cpu_host.iter().zip(gpu_host.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cpu/gpu randn diverged at i={}: cpu={}, gpu={}",
                i,
                a.to_f32(),
                b.to_f32()
            );
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod opbackend_dispatch_tests {
    //! Integration tests that exercise the new diffusion ops through the
    //! `OpBackend` trait dispatch (not just the kernel modules directly).
    //! Catches issues where the trait wiring forgets a method.

    use crate::domain::tensor::Tensor;
    use crate::domain::ports::{CoreOps, DiffusionOps};
    use crate::infrastructure::cuda::Cuda;
    use half::bf16;

    #[test]
    fn opbackend_apply_rope_interleaved_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (4usize, 2usize, 8usize);
        let half = d / 2;
        let x_host: Vec<f32> = (0..seq * h * d).map(|i| (i as f32) * 0.1).collect();
        let cos_host: Vec<f32> = vec![0.9; seq * half];
        let sin_host: Vec<f32> = vec![0.1; seq * half];
        let mut x: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&x_host, [seq, h, d], &cuda).unwrap();
        let cos_t: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&cos_host, [seq, half], &cuda).unwrap();
        let sin_t: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&sin_host, [seq, half], &cuda).unwrap();
        Cuda::apply_rope_interleaved(&mut x, &cos_t, &sin_t, d).unwrap();
        let got = x.to_host_vec().unwrap();
        // Sanity: values changed.
        assert_ne!(got[0], x_host[0]);
        // First rotated pair: x'[0] = a*cos - b*sin, x'[1] = a*sin + b*cos.
        let (a, b) = (x_host[0], x_host[1]);
        assert!((got[0] - (a * 0.9 - b * 0.1)).abs() < 1e-5);
        assert!((got[1] - (a * 0.1 + b * 0.9)).abs() < 1e-5);
    }

    #[test]
    fn opbackend_concat_seq_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let a: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &(0..8).map(|i| bf16::from_f32(i as f32)).collect::<Vec<_>>(),
            [2, 4],
            &cuda,
        )
        .unwrap();
        let b: Tensor<bf16, Cuda> = Tensor::from_host_slice(
            &(0..12)
                .map(|i| bf16::from_f32(-(i as f32)))
                .collect::<Vec<_>>(),
            [3, 4],
            &cuda,
        )
        .unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([5, 4], &cuda).unwrap();
        Cuda::concat_seq(&a, &b, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(got[0].to_f32(), 0.0);
        assert_eq!(got[7].to_f32(), 7.0);
        assert_eq!(got[8].to_f32(), 0.0); // start of b
        assert_eq!(got[19].to_f32(), -11.0);
    }

    #[test]
    fn opbackend_sdpa_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let (seq, h, d) = (3usize, 2usize, 4usize);
        let scale = 1.0 / (d as f32).sqrt();
        let q: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(1)).unwrap();
        let k: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(2)).unwrap();
        let v: Tensor<f32, Cuda> = Tensor::randn([seq, h, d], &cuda, Some(3)).unwrap();
        let mut out: Tensor<f32, Cuda> = Tensor::zeros([seq, h, d], &cuda).unwrap();
        Cuda::sdpa(&q, &k, &v, &mut out, h, h, d, scale).unwrap();
        // Output is a finite, nonzero tensor.
        let got = out.to_host_vec().unwrap();
        assert!(got.iter().all(|x| x.is_finite()));
        assert!(got.iter().any(|x| x.abs() > 1e-6));
    }

    #[test]
    fn opbackend_pad_with_token_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [1, 4], &cuda).unwrap();
        let pad: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[7.0, 7.0, 7.0, 7.0], [4], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([3, 4], &cuda).unwrap();
        Cuda::pad_with_token(&src, &pad, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(&got[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert!(got[4..].iter().all(|&x| x == 7.0));
    }

    #[test]
    fn opbackend_cast_dtype_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [4], &cuda).unwrap();
        let mut dst: Tensor<bf16, Cuda> = Tensor::zeros([4], &cuda).unwrap();
        Cuda::cast_dtype(&src, &mut dst).unwrap();
        let got: Vec<f32> = dst
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn opbackend_silu_and_tanh_dispatch() {
        let cuda = Cuda::new(0).unwrap();
        let mut s: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[1.0, 2.0, -1.0], [3], &cuda).unwrap();
        Cuda::silu_inplace_diff(&mut s).unwrap();
        let s_got = s.to_host_vec().unwrap();
        for (i, &x) in [1.0_f32, 2.0, -1.0].iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!((s_got[i] - expected).abs() < 1e-5);
        }
        let mut t: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[0.5, 1.0, -1.0], [3], &cuda).unwrap();
        Cuda::tanh_inplace(&mut t).unwrap();
        let t_got = t.to_host_vec().unwrap();
        for (i, &x) in [0.5_f32, 1.0, -1.0].iter().enumerate() {
            assert!((t_got[i] - x.tanh()).abs() < 1e-5);
        }
    }

    #[test]
    fn opbackend_scalar_mul_from_dev_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let mut x: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[1.0, 2.0, 3.0], [3], &cuda).unwrap();
        let scalar: Tensor<f32, Cuda> = Tensor::from_host_slice(&[2.5_f32], [1], &cuda).unwrap();
        Cuda::scalar_mul_inplace_from_dev(&mut x, &scalar).unwrap();
        let got = x.to_host_vec().unwrap();
        assert!((got[0] - 2.5).abs() < 1e-5);
        assert!((got[1] - 5.0).abs() < 1e-5);
        assert!((got[2] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn opbackend_split_cols_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let src_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let src: Tensor<f32, Cuda> = Tensor::from_host_slice(&src_host, [2, 3], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([2, 2], &cuda).unwrap();
        Cuda::split_cols(&src, &mut dst, 2, 3, 1, 2).unwrap();
        let got = dst.to_host_vec().unwrap();
        assert_eq!(got, vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn opbackend_broadcast_add_dispatches() {
        let cuda = Cuda::new(0).unwrap();
        let mut x: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2], &cuda).unwrap();
        let bias: Tensor<f32, Cuda> = Tensor::from_host_slice(&[10.0, 20.0], [2], &cuda).unwrap();
        Cuda::broadcast_add_inplace(&mut x, &bias).unwrap();
        let got = x.to_host_vec().unwrap();
        assert_eq!(got, vec![11.0, 22.0, 13.0, 24.0]);
    }
}
