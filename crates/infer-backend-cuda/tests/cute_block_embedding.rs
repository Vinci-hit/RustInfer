//! GPU execution tests; require --features cute-dsl -- --ignored --test-threads=1.
#![cfg(feature = "cute-dsl")]
use infer_backend_cpu::block_quant::decode_row;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope};
use infer_core::{
    dtype::{Dtype, quant::BlockQuantFormat as F},
    exec::ExecScope,
    ports::{MathOps, OpError},
    quantized::{BlockQuantLayout, BlockQuantView, BlockQuantWeight},
    tensor::Tensor,
    types::Strides,
};

fn scope() -> CudaScope {
    let s = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024 * 1024,
            pool_retain_bytes: 4 * 1024 * 1024,
        },
    )
    .unwrap()
    .scope();
    assert!(s.device().config.cute_dsl_available());
    s
}

fn fixture(f: F) -> Vec<u8> {
    let file = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../infer-backend-cpu/tests/fixtures/block_quant/{f:?}.bin"
        )),
    )
    .unwrap();
    let b = f.layout();
    (0..64)
        .flat_map(|i| {
            file[4 + i * (b.bytes + 4 * b.elements)..4 + i * (b.bytes + 4 * b.elements) + b.bytes]
                .iter()
                .copied()
        })
        .collect()
}

fn check<T: Dtype>(s: &CudaScope) {
    for &f in F::ALL {
        let b = f.layout();
        let bytes = fixture(f);
        // Two blocks per row exercise block byte strides, including unaligned
        // half scales, and a nonzero row offset in a shared GPU allocation.
        let layout = BlockQuantLayout::new(f, 32, b.elements * 2).unwrap();
        let view = BlockQuantView::new(layout, &bytes).unwrap();
        // Put the first encoded block at an odd byte address as well: no
        // FP16/u16 alignment may be assumed by the byte decoder.
        let mut padded = vec![0xa5];
        padded.extend_from_slice(&bytes);
        padded.push(0x5a);
        let allocation = Tensor::from_host_slice(&padded, [padded.len()], s.device()).unwrap();
        let parent =
            BlockQuantWeight::try_new(layout, allocation.narrow(0, 1, bytes.len()).unwrap())
                .unwrap();
        for sliced in [false, true] {
            let (w, v) = if sliced {
                (
                    parent.slice_rows(1..31).unwrap(),
                    view.slice_rows(1..31).unwrap(),
                )
            } else {
                (parent.clone(), view)
            };
            let [n, k] = v.layout().shape();
            let ids: Vec<i32> = (0..n)
                .rev()
                .chain([0, n - 1, 0])
                .map(|i| i as i32)
                .collect();
            let mut padded_ids = vec![-1];
            for &id in &ids {
                padded_ids.extend([id, -1]);
            }
            let storage =
                Tensor::from_host_slice(&padded_ids, [padded_ids.len()], s.device()).unwrap();
            let input = storage.view_raw([ids.len()].into(), Strides::from_slice(&[2]), 1, false);
            // Strides on both axes, plus a nonzero allocation offset.
            let pitch = k * 2 + 3;
            let output = Tensor::<T, Cuda>::from_host_slice(
                &vec![T::write_f64(123.0); ids.len() * pitch],
                [ids.len(), pitch],
                s.device(),
            )
            .unwrap();
            let mut dst = output.view_raw(
                [ids.len(), k].into(),
                Strides::from_slice(&[pitch, 2]),
                1,
                false,
            );
            Cuda::embedding_block_quant(s, &w, &input, &mut dst).unwrap();
            s.synchronize().unwrap();
            let actual = output.to_host_vec().unwrap();
            let mut expected = vec![0.0; k];
            for (row, &id) in ids.iter().enumerate() {
                decode_row(v, id as usize, &mut expected).unwrap();
                for col in 0..pitch {
                    let expected = if col >= 1 && col < 1 + 2 * k && col % 2 == 1 {
                        T::read_f64(&T::write_f64(expected[(col - 1) / 2] as f64))
                    } else {
                        123.0
                    };
                    let actual = T::read_f64(&actual[row * pitch + col]);
                    assert_eq!(
                        actual,
                        expected,
                        "{f:?}/{:?}, sliced={sliced}, token {id} column {col}",
                        T::ID
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn f32_all_formats_match_cpu() {
    check::<f32>(&scope());
}
#[test]
#[ignore = "requires CUDA"]
fn f16_all_formats_match_cpu() {
    check::<half::f16>(&scope());
}
#[test]
#[ignore = "requires CUDA"]
fn bf16_all_formats_match_cpu() {
    check::<half::bf16>(&scope());
}

#[test]
#[ignore = "requires CUDA"]
fn invalid_inputs_leave_output_untouched_and_capture_is_rejected() {
    let s = scope();
    let bytes = fixture(F::Q3_K);
    let l = BlockQuantLayout::new(F::Q3_K, 64, 256).unwrap();
    let w =
        BlockQuantWeight::from_host(BlockQuantView::new(l, &bytes).unwrap(), s.device()).unwrap();
    let mut out = Tensor::from_host_slice(&[123.0f32; 512], [2, 256], s.device()).unwrap();
    for invalid in [-1, 64, i32::MAX] {
        let ids = Tensor::from_host_slice(&[0i32, invalid], [2], s.device()).unwrap();
        assert!(matches!(
            Cuda::embedding_block_quant(&s, &w, &ids, &mut out),
            Err(OpError::Shape(_))
        ));
        assert_eq!(out.to_host_vec().unwrap(), [123.0; 512]);
    }
    let ids = Tensor::from_host_slice(&[1i32, 2], [2], s.device()).unwrap();
    let bad = ids.view_contiguous([1, 2].into()).unwrap();
    assert!(Cuda::embedding_block_quant(&s, &w, &bad, &mut out).is_err());
    let mut wrong = out.narrow(1, 0, 255).unwrap();
    assert!(Cuda::embedding_block_quant(&s, &w, &ids, &mut wrong).is_err());
    let mut overlapping = out.view_raw([2, 256].into(), Strides::from_slice(&[128, 1]), 0, false);
    assert!(Cuda::embedding_block_quant(&s, &w, &ids, &mut overlapping).is_err());
    let mut aliased_weight = Tensor::<f32, Cuda>::from_raw_parts(
        std::sync::Arc::clone(w.bytes().storage()),
        [2, 256].into(),
        Strides::from_slice(&[256, 1]),
        0,
        true,
    );
    assert!(Cuda::embedding_block_quant(&s, &w, &ids, &mut aliased_weight).is_err());
    let mut integer = Tensor::<i32, Cuda>::zeros([2, 256], s.device()).unwrap();
    assert!(matches!(
        Cuda::embedding_block_quant(&s, &w, &ids, &mut integer),
        Err(OpError::Unsupported { .. })
    ));
    s.graph_capture_begin().unwrap();
    assert!(matches!(
        Cuda::embedding_block_quant(&s, &w, &ids, &mut out),
        Err(OpError::Unsupported { .. })
    ));
    s.graph_capture_abort().unwrap();
    assert_eq!(out.to_host_vec().unwrap(), [123.0; 512]);
    // Eager use must remain healthy after the rejected capture.
    Cuda::embedding_block_quant(&s, &w, &ids, &mut out).unwrap();
    s.synchronize().unwrap();
    let empty = Tensor::<i32, Cuda>::zeros([0], s.device()).unwrap();
    let mut out = Tensor::<f32, Cuda>::zeros([0, 256], s.device()).unwrap();
    Cuda::embedding_block_quant(&s, &w, &empty, &mut out).unwrap();
}

#[test]
#[ignore = "requires CUDA"]
fn independent_contexts_and_broadcast_ids() {
    let first = scope();
    let bytes = fixture(F::Q8_0);
    let l = BlockQuantLayout::new(F::Q8_0, 64, 32).unwrap();
    let view = BlockQuantView::new(l, &bytes).unwrap();
    let w = BlockQuantWeight::from_host(view, first.device()).unwrap();
    let ids = Tensor::from_host_slice(&[6i32; 3], [3], first.device()).unwrap();
    let broadcast = ids.view_raw([3].into(), Strides::from_slice(&[0]), 0, false);
    let mut out = Tensor::<f32, Cuda>::zeros([3, 32], first.device()).unwrap();
    {
        let second = scope();
        let foreign = Tensor::from_host_slice(&[0i32; 3], [3], second.device()).unwrap();
        assert!(Cuda::embedding_block_quant(&first, &w, &foreign, &mut out).is_err());
        let foreign_w = BlockQuantWeight::from_host(view, second.device()).unwrap();
        assert!(Cuda::embedding_block_quant(&first, &foreign_w, &ids, &mut out).is_err());
        let mut foreign_out = Tensor::<f32, Cuda>::zeros([3, 32], second.device()).unwrap();
        assert!(Cuda::embedding_block_quant(&first, &w, &ids, &mut foreign_out).is_err());
    }
    Cuda::embedding_block_quant(&first, &w, &broadcast, &mut out).unwrap();
    let mut expected = vec![0.0; 32];
    decode_row(view, 6, &mut expected).unwrap();
    assert_eq!(out.to_host_vec().unwrap(), expected.repeat(3));
    let storage = Tensor::<f32, Cuda>::zeros([32, 3], first.device()).unwrap();
    let mut transposed = storage.view_raw([3, 32].into(), Strides::from_slice(&[1, 3]), 0, false);
    Cuda::embedding_block_quant(&first, &w, &broadcast, &mut transposed).unwrap();
    let values = storage.to_host_vec().unwrap();
    for col in 0..32 {
        assert_eq!(&values[col * 3..col * 3 + 3], &[expected[col]; 3]);
    }
}
