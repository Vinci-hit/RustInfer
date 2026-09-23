use infer_backend_cpu::{
    Cpu,
    block_quant::{decode_block, decode_row},
};
use infer_core::{
    dtype::{Dtype, quant::BlockQuantFormat as F},
    exec::HostScope,
    ports::MathOps,
    quantized::{BlockQuantLayout, BlockQuantView, BlockQuantWeight},
    tensor::Tensor,
    types::Strides,
};

fn fixture(f: F) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/fixtures/block_quant/{f:?}.bin")),
    )
    .unwrap()
}

fn matrix_values<T: Dtype>(t: &Tensor<T, Cpu>) -> Vec<T> {
    let s = t.strides().as_slice();
    (0..t.shape()[0])
        .flat_map(|i| {
            (0..t.shape()[1]).map(move |j| {
                // SAFETY: host tensor, valid indices respecting its layout.
                unsafe { *t.data_ptr().add(i * s[0] + j * s[1]) }
            })
        })
        .collect()
}

fn case(f: F, case: usize) -> (Vec<u8>, Vec<f32>) {
    let file = fixture(f);
    let b = f.layout();
    let start = 4 + case * (b.bytes + b.elements * 4);
    (
        file[start..start + b.bytes].to_vec(),
        file[start + b.bytes..start + b.bytes + b.elements * 4]
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
    )
}

fn weight(f: F, n: usize, blocks: usize) -> (BlockQuantWeight<Cpu>, Vec<f32>) {
    let mut encoded = vec![];
    let mut expected = vec![];
    for i in 0..n * blocks {
        let (b, y) = case(f, (i + 5) % 64);
        encoded.extend(b);
        expected.extend(y);
    }
    let l = BlockQuantLayout::new(f, n, blocks * f.layout().elements).unwrap();
    (
        BlockQuantWeight::from_host(BlockQuantView::new(l, &encoded).unwrap(), &Cpu).unwrap(),
        expected,
    )
}

#[test]
fn all_formats_match_encode_side_vectors() {
    let mut count = 0;
    for &f in F::ALL {
        let file = fixture(f);
        let b = f.layout();
        assert_eq!(u32::from_le_bytes(file[..4].try_into().unwrap()), 64);
        assert_eq!(file.len(), 4 + 64 * (b.bytes + b.elements * 4));
        for id in 0..64 {
            let (src, expected) = case(f, id);
            let mut out = vec![f32::NAN; b.elements];
            decode_block(f, &src, &mut out).unwrap();
            for (i, (&x, &y)) in out.iter().zip(&expected).enumerate() {
                assert_eq!(x, y, "{f:?} case {id}, element {i}");
            }
            count += 1;
        }
    }
    assert_eq!(count, 832);
}

#[test]
fn row_slices_multiple_blocks_and_invalid_lengths() {
    for &f in F::ALL {
        let (w, expected) = weight(f, 3, 2);
        let k = w.layout().shape()[1];
        let bytes = w.bytes().to_host_vec().unwrap();
        let v = BlockQuantView::new(*w.layout(), &bytes)
            .unwrap()
            .slice_rows(1..3)
            .unwrap();
        let mut out = vec![123.0; k];
        decode_row(v, 1, &mut out).unwrap();
        assert_eq!(out, &expected[2 * k..3 * k]);
        out.fill(123.0);
        assert!(decode_row(v, 2, &mut out).is_err());
        assert!(decode_row(v, 0, &mut out[..k - 1]).is_err());
        let b = f.layout();
        assert!(decode_block(f, &bytes[..b.bytes - 1], &mut out[..b.elements]).is_err());
        assert!(decode_block(f, &bytes[..b.bytes], &mut out[..b.elements - 1]).is_err());
        assert!(out.iter().all(|&x| x == 123.0));
    }
}

#[test]
fn scale_ieee_values_and_unaligned_little_endian_bytes() {
    for &f in F::ALL {
        let (b, _) = case(f, 6);
        let mut unaligned = vec![0xff];
        unaligned.extend(b);
        let mut out = vec![0.0; f.layout().elements];
        decode_block(f, &unaligned[1..], &mut out).unwrap();
        assert_eq!(out, case(f, 6).1);
        let offset = match f {
            F::Q2_K => 80,
            F::Q3_K => 108,
            F::Q6_K => 208,
            _ => 0,
        };
        unaligned[offset + 1..offset + 3].copy_from_slice(&0x7e00u16.to_le_bytes());
        decode_block(f, &unaligned[1..], &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_nan()));
    }
    let mut src = vec![1; 34];
    src[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
    let mut out = [0.0; 32];
    decode_block(F::Q8_0, &src, &mut out).unwrap();
    assert!(out.iter().all(|&v| v == f32::INFINITY));
}

fn exercise_ops<T: Dtype>() {
    let scope = HostScope::new(Cpu);
    for &f in F::ALL {
        let (w, expected) = weight(f, 3, 2);
        let k = w.layout().shape()[1];
        // Input and destination both have padding and nonzero offsets.
        let physical = Tensor::<T, Cpu>::from_host_slice(
            &vec![T::write_f64(0.25); 2 * (k + 3)],
            [2, k + 3],
            &Cpu,
        )
        .unwrap();
        let x = physical.narrow(1, 1, k).unwrap();
        let bias = Tensor::<T, Cpu>::from_host_slice(&[T::write_f64(0.125); 3], [3], &Cpu).unwrap();
        let parent =
            Tensor::<T, Cpu>::from_host_slice(&[T::write_f64(123.0); 14], [2, 7], &Cpu).unwrap();
        let mut y = parent.narrow(1, 2, 3).unwrap();
        Cpu::matmul_block_quant(&scope, &x, &w, Some(&bias), &mut y).unwrap();
        let actual = matrix_values(&y);
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0f32;
                for &v in &expected[j * k..(j + 1) * k] {
                    sum += 0.25 * v;
                }
                sum += 0.125;
                assert_eq!(
                    T::read_f64(&actual[i * 3 + j]),
                    T::read_f64(&T::write_f64(sum as f64)),
                    "{f:?}"
                );
            }
        }
        let full = parent.to_host_vec().unwrap();
        for i in [0, 1, 5, 6, 7, 8, 12, 13] {
            assert_eq!(T::read_f64(&full[i]), 123.0);
        }

        // Noncontiguous ID vector, repeats, row-sliced weights and columns.
        let ids = Tensor::from_host_slice(&[1i32, 99, 0, 99, 1, 99], [6], &Cpu).unwrap();
        let ids = ids.view_raw([3].into(), Strides::from_slice(&[2]), 0, false);
        let sub = w.slice_rows(1..3).unwrap();
        let parent = Tensor::<T, Cpu>::from_host_slice(
            &vec![T::write_f64(123.0); 3 * (k + 2)],
            [3, k + 2],
            &Cpu,
        )
        .unwrap();
        let mut y = parent.narrow(1, 1, k).unwrap();
        Cpu::embedding_block_quant(&scope, &sub, &ids, &mut y).unwrap();
        let actual = matrix_values(&y);
        for (row, token) in [2, 1, 2].into_iter().enumerate() {
            for j in 0..k {
                assert_eq!(
                    T::read_f64(&actual[row * k + j]),
                    T::read_f64(&T::write_f64(expected[token * k + j] as f64)),
                    "{f:?}"
                );
            }
        }
        let full = parent.to_host_vec().unwrap();
        for row in 0..3 {
            for col in [0, k + 1] {
                assert_eq!(T::read_f64(&full[row * (k + 2) + col]), 123.0);
            }
        }
    }
}

#[test]
fn operators_f32() {
    exercise_ops::<f32>();
}
#[test]
fn operators_f16() {
    exercise_ops::<half::f16>();
}
#[test]
fn operators_bf16() {
    exercise_ops::<half::bf16>();
}

#[test]
fn operators_reject_bad_inputs_before_writing() {
    let scope = HostScope::new(Cpu);
    let (w, _) = weight(F::Q8_0, 2, 1);
    let x = Tensor::<f32, Cpu>::zeros([2, 32], &Cpu).unwrap();
    let mut y = Tensor::<f32, Cpu>::from_host_slice(&[123.0; 4], [2, 2], &Cpu).unwrap();
    let bias = Tensor::<f32, Cpu>::zeros([1], &Cpu).unwrap();
    assert!(Cpu::matmul_block_quant(&scope, &x, &w, Some(&bias), &mut y).is_err());
    let wrong = Tensor::<f32, Cpu>::zeros([2, 31], &Cpu).unwrap();
    assert!(Cpu::matmul_block_quant(&scope, &wrong, &w, None, &mut y).is_err());
    assert_eq!(y.to_host_vec().unwrap(), [123.0; 4]);
    let mut y = Tensor::<f32, Cpu>::from_host_slice(&[123.0; 64], [2, 32], &Cpu).unwrap();
    for id in [-1, 2, i32::MAX] {
        let ids = Tensor::<i32, Cpu>::from_host_slice(&[0, id], [2], &Cpu).unwrap();
        assert!(Cpu::embedding_block_quant(&scope, &w, &ids, &mut y).is_err());
        assert_eq!(y.to_host_vec().unwrap(), [123.0; 64]);
    }
    let mut alias = x.narrow(1, 0, 2).unwrap();
    assert!(Cpu::matmul_block_quant(&scope, &x, &w, None, &mut alias).is_err());
    let xi = Tensor::<i32, Cpu>::zeros([2, 32], &Cpu).unwrap();
    let mut yi = Tensor::<i32, Cpu>::zeros([2, 2], &Cpu).unwrap();
    assert!(Cpu::matmul_block_quant(&scope, &xi, &w, None, &mut yi).is_err());
    let empty = Tensor::<f32, Cpu>::zeros([0, 32], &Cpu).unwrap();
    let mut y = Tensor::<f32, Cpu>::zeros([0, 2], &Cpu).unwrap();
    Cpu::matmul_block_quant(&scope, &empty, &w, None, &mut y).unwrap();
    let empty = Tensor::<i32, Cpu>::zeros([0], &Cpu).unwrap();
    let mut y = Tensor::<f32, Cpu>::zeros([0, 32], &Cpu).unwrap();
    Cpu::embedding_block_quant(&scope, &w, &empty, &mut y).unwrap();
}
