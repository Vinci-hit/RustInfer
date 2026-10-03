use super::*;
use crate::components::linear::LinearWeight;
use crate::infrastructure::cpu::Cpu;
use safetensors::{Dtype as StDtype, tensor::TensorView};
use std::collections::BTreeMap;

type Entry = (StDtype, Vec<usize>, Vec<u8>);

fn reader(tensors: &BTreeMap<String, Entry>) -> SafetensorsReader {
    let views: BTreeMap<_, _> = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let path = std::env::temp_dir().join(format!(
        "rustinfer-int4-{}-{:?}.safetensors",
        std::process::id(),
        std::thread::current().id()
    ));
    safetensors::tensor::serialize_to_file(views, None, &path).unwrap();
    let reader = SafetensorsReader::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    reader
}

fn fixture(symmetric: bool, group: usize) -> BTreeMap<String, Entry> {
    let mut tensors = BTreeMap::new();
    for (part, n) in [("q", 11), ("k", 5), ("v", 8)] {
        let k = 256;
        let groups = k / group;
        let packed: Vec<i32> = (0..n * k / 8)
            .map(|i| {
                let row = i / (k / 8);
                (0..8).fold(0, |word, bit| {
                    word | (((row + i * 8 + bit) % 16) as i32) << (bit * 4)
                })
            })
            .collect();
        let mut zeros = vec![0i32; n.div_ceil(8) * groups];
        for row in 0..n {
            for g in 0..groups {
                zeros[(row / 8) * groups + g] |= (((row + g) % 16) as i32) << ((row % 8) * 4);
            }
        }
        tensors.insert(
            format!("{part}.weight_packed"),
            (
                StDtype::I32,
                vec![n, k / 8],
                packed.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        );
        tensors.insert(
            format!("{part}.weight_scale"),
            (
                StDtype::F32,
                vec![n, groups],
                (0..n * groups)
                    .flat_map(|i| (((i % groups + 1) as f32) / 64.).to_le_bytes())
                    .collect(),
            ),
        );
        tensors.insert(
            format!("{part}.weight_shape"),
            (
                StDtype::I64,
                vec![2],
                [n as i64, k as i64]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
        );
        if !symmetric {
            tensors.insert(
                format!("{part}.weight_zero_point"),
                (
                    StDtype::I32,
                    vec![n.div_ceil(8), groups],
                    zeros.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            );
        }
    }
    tensors
}

fn scheme(symmetric: bool, group: usize) -> QuantScheme {
    QuantScheme {
        group,
        symmetry: if symmetric {
            Symmetry::Symmetric
        } else {
            Symmetry::Asymmetric
        },
        ..QuantScheme::AWQ_INT4_G128
    }
}

#[test]
fn int4_fusion_repacks_unaligned_zero_points_and_supports_symmetric() {
    for symmetric in [false, true] {
        let reader = reader(&fixture(symmetric, 128));
        let loader = WeightLoader::new(&reader);
        let linear = loader
            .load_int4_parts::<f32, Cpu>(
                &[("q", 11), ("k", 5), ("v", 8)],
                256,
                scheme(symmetric, 128),
                &Cpu,
            )
            .unwrap();
        let LinearWeight::Awq {
            packed,
            zeros,
            scales,
            ..
        } = linear.weight
        else {
            panic!("expanded INT4 weight")
        };
        assert_eq!(packed.shape().as_slice(), &[24, 32]);
        assert_eq!(scales.shape().as_slice(), &[24, 2]);
        let actual = zeros.to_host_vec().unwrap();
        let mut row = 0;
        for n in [11, 5, 8] {
            for local in 0..n {
                for g in 0..2 {
                    let expected = if symmetric { 8 } else { (local + g) % 16 };
                    assert_eq!(
                        (actual[row / 8 * 2 + g] >> (row % 8 * 4)) & 15,
                        expected as i32
                    );
                }
                row += 1;
            }
        }
    }
}

#[test]
fn int4_loader_rejects_malformed_shapes_and_missing_zero_points() {
    for suffix in [
        "weight_packed",
        "weight_scale",
        "weight_zero_point",
        "weight_shape",
    ] {
        let mut tensors = fixture(false, 128);
        tensors.remove(&format!("q.{suffix}"));
        if suffix == "weight_shape" {
            tensors.insert(
                "q.weight_shape".into(),
                (
                    StDtype::I64,
                    vec![2],
                    [11i64, 255].iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            );
        }
        let reader = reader(&tensors);
        assert!(
            WeightLoader::new(&reader)
                .load_int4_parts::<f32, Cpu>(&[("q", 11)], 256, scheme(false, 128), &Cpu)
                .is_err()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device"]
fn int4_gpu_fused_projections_match_independent_dequantization() {
    use half::bf16;
    use infer_backend_cuda::{Cuda, CudaMemoryPlan};
    use infer_core::exec::ExecScope;
    use infer_core::ports::MathOps;
    let scope = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope();
    for symmetric in [false, true] {
        for group in [8, 128, 256] {
            let reader = reader(&fixture(symmetric, group));
            let linear = WeightLoader::new(&reader)
                .load_int4_parts::<bf16, Cuda>(
                    &[("q", 11), ("k", 5), ("v", 8)],
                    256,
                    scheme(symmetric, group),
                    scope.device(),
                )
                .unwrap();
            let LinearWeight::Awq {
                packed,
                zeros,
                scales,
                scheme,
            } = linear.weight
            else {
                unreachable!()
            };
            for m in [1, 3] {
                let host: Vec<_> = (0..m * 256)
                    .map(|i| bf16::from_f32(((i % 5) as f32 - 2.) / 16.))
                    .collect();
                let input = Tensor::from_host_slice(&host, [m, 256], scope.device()).unwrap();
                let mut output = Tensor::<bf16, Cuda>::zeros([m, 24], scope.device()).unwrap();
                Cuda::matmul_quant(
                    &scope,
                    &input,
                    &packed,
                    &mut output,
                    &scales,
                    Some(&zeros),
                    &scheme,
                )
                .unwrap();
                let actual = output.to_host_vec().unwrap();
                for batch in 0..m {
                    let mut row = 0;
                    for n in [11, 5, 8] {
                        for local in 0..n {
                            let expected: f32 = (0..256)
                                .map(|col| {
                                    let q = ((local + col) % 16) as f32;
                                    let zero = if symmetric {
                                        8.
                                    } else {
                                        ((local + col / group) % 16) as f32
                                    };
                                    let scale = (col / group + 1) as f32 / 64.;
                                    host[batch * 256 + col].to_f32() * (q - zero) * scale
                                })
                                .sum();
                            let value = actual[batch * 24 + row].to_f32();
                            assert!(
                                (value - expected).abs() < 0.025 + 0.025 * expected.abs(),
                                "m={m}, row={row}, group={group}: {value} vs {expected}"
                            );
                            row += 1;
                        }
                    }
                }
                assert!(
                    Cuda::matmul_quant(
                        &scope,
                        &input,
                        &packed,
                        &mut output,
                        &scales,
                        None,
                        &scheme
                    )
                    .is_err()
                );
            }
        }
    }
}
