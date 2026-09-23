use std::path::Path;
use std::sync::Arc;

use infer_backend_cpu::Cpu;
use infer_core::component::Hidden;
use infer_core::dtype::quant::BlockQuantFormat;
use infer_core::error::OpError;
use infer_core::exec::{HostScope, RankPair, StepCtx, TopologyShape};
use infer_core::plan::{BatchKind, BatchPlan};
use infer_core::quantized::{BlockQuantLayout, BlockQuantView, BlockQuantWeight};
use infer_core::tensor::Tensor;
use infer_worker::components::block_quant_projection::BlockQuantProjection;
use infer_worker::components::embed::{Embed, EmbeddingParallelism};
use infer_worker::components::linear::{Linear, LinearParallelism, LinearWeight};
use infer_worker::infrastructure::io::gguf::GgufReader;
use infer_worker::models::gguf_weights::block_quant_view;

fn weight(format: BlockQuantFormat, n: usize, k: usize) -> BlockQuantWeight<Cpu> {
    let layout = BlockQuantLayout::new(format, n, k).unwrap();
    let data: Vec<_> = (0..layout.byte_len()).map(|i| i as u8).collect();
    BlockQuantWeight::from_host(BlockQuantView::new(layout, &data).unwrap(), &Cpu).unwrap()
}

fn plan() -> BatchPlan {
    BatchPlan {
        kind: BatchKind::DecodeOnly,
        num_tokens: 2,
        batch: 2,
        q_lens: vec![1, 1],
        kv_lens: vec![1, 1],
        seq_positions: vec![0, 0],
        rope_positions: vec![0, 0],
        max_blocks_per_seq: 1,
        block_size: 1,
        total_q_tiles: 0,
    }
}

#[test]
fn storage_subviews_survive_source_and_parent_and_require_exact_bytes() {
    for &format in BlockQuantFormat::ALL {
        let original = weight(format, 5, format.layout().elements * 2);
        let reference = original.bytes().to_host_vec().unwrap();
        let stride = original.layout().row_bytes();
        let cloned = original.clone();
        let sub = original.slice_rows(1..5).unwrap().slice_rows(1..3).unwrap();
        assert!(Arc::ptr_eq(
            original.bytes().storage(),
            sub.bytes().storage()
        ));
        assert!(Arc::ptr_eq(
            original.bytes().storage(),
            cloned.bytes().storage()
        ));
        assert_eq!(sub.bytes().offset_elems(), stride * 2);
        drop(original);
        drop(cloned);
        assert_eq!(
            sub.bytes().to_host_vec().unwrap(),
            &reference[stride * 2..stride * 4]
        );
        let len = sub.layout().byte_len();
        let wrong_shape = Tensor::<u8, Cpu>::zeros([1, len], &Cpu).unwrap();
        assert!(BlockQuantWeight::try_new(*sub.layout(), wrong_shape).is_err());
        assert!(
            BlockQuantWeight::try_new(
                *sub.layout(),
                Tensor::<u8, Cpu>::zeros([len - 1], &Cpu).unwrap()
            )
            .is_err()
        );
        let physical = Tensor::<u8, Cpu>::zeros([len * 2], &Cpu).unwrap();
        let strided = physical.view_raw(
            [len].into(),
            infer_core::types::Strides::from_slice(&[2]),
            0,
            false,
        );
        assert!(BlockQuantWeight::try_new(*sub.layout(), strided).is_err());
    }
}

#[test]
fn mixed_qkv_preserves_formats_storage_and_multi_token_output_strides() {
    let parts = vec![
        weight(BlockQuantFormat::IQ4_NL, 12288, 5120),
        weight(BlockQuantFormat::Q4_K, 1024, 5120),
        weight(BlockQuantFormat::Q5_K, 1024, 5120),
    ];
    let originals: Vec<_> = parts
        .iter()
        .map(|w| Arc::clone(w.bytes().storage()))
        .collect();
    let projection = BlockQuantProjection::try_new(parts).unwrap();
    assert_eq!(projection.shape(), [14336, 5120]);
    let output = Tensor::<f32, Cpu>::zeros([2, 14336], &Cpu).unwrap();
    for ((range, part), storage) in projection.parts().zip(originals) {
        assert!(Arc::ptr_eq(&storage, part.bytes().storage()));
        let output_part = output.narrow(1, range.start, range.len()).unwrap();
        assert_eq!(output_part.strides().as_slice(), [14336, 1]);
        assert!(!output_part.is_contiguous());
        assert_eq!(output_part.offset_elems(), range.start);
    }
    let formats: Vec<_> = projection
        .parts()
        .map(|(_, p)| p.layout().format())
        .collect();
    assert_eq!(
        formats,
        [
            BlockQuantFormat::IQ4_NL,
            BlockQuantFormat::Q4_K,
            BlockQuantFormat::Q5_K
        ]
    );
    assert!(BlockQuantProjection::<Cpu>::try_new(vec![]).is_err());
    assert!(
        BlockQuantProjection::try_new(vec![
            weight(BlockQuantFormat::Q3_K, 1, 256),
            weight(BlockQuantFormat::Q3_K, 1, 512)
        ])
        .is_err()
    );
}

#[test]
fn block_components_fail_before_writing_and_tied_weights_share_storage() {
    let w = weight(BlockQuantFormat::Q3_K, 3, 256);
    let embedding = Embed::<f32, Cpu>::from_block_quant(w.clone());
    assert!(embedding.as_dense().is_none());
    assert!(matches!(
        embedding.require_dense(),
        Err(OpError::Unsupported { .. })
    ));
    let shared = embedding.shallow_clone();
    let linear = shared.shared_linear().unwrap();
    assert_eq!(linear.weight.logical_shape().unwrap(), [3, 256]);
    assert!(linear.weight.as_dense().is_none());
    let LinearWeight::BlockQuant(projection) = &linear.weight else {
        panic!("quantized")
    };
    assert!(Arc::ptr_eq(
        w.bytes().storage(),
        projection.parts().next().unwrap().1.bytes().storage()
    ));

    let scope = HostScope::new(Cpu);
    let plan = plan();
    let ctx = StepCtx::new(&scope, &plan);
    let input = Tensor::<f32, Cpu>::zeros([2, 255], &Cpu).unwrap();
    let mut output = Tensor::from_host_slice(&[123.0f32; 6], [2, 3], &Cpu).unwrap();
    assert!(matches!(
        linear.forward(&input, &mut output, &ctx),
        Err(OpError::Shape(_))
    ));
    assert_eq!(output.to_host_vec().unwrap(), [123.0; 6]);

    let ids = Tensor::from_host_slice(&[0i32, 3], [2], &Cpu).unwrap();
    let mut hidden = Hidden {
        stream: Tensor::from_host_slice(&[123.0f32; 512], [2, 256], &Cpu).unwrap(),
        pending: None,
    };
    assert!(matches!(
        embedding.forward(&ids, &mut hidden, &ctx),
        Err(OpError::Shape(_))
    ));
    assert_eq!(hidden.stream.to_host_vec().unwrap(), [123.0; 512]);

    let tp = RankPair { rank: 0, size: 2 };
    let scope2 = HostScope::new(Cpu).with_topology(TopologyShape {
        tp,
        ..TopologyShape::SINGLE
    });
    let ctx2 = StepCtx::new(&scope2, &plan);
    let linear = linear.with_parallelism(LinearParallelism::Column {
        tp,
        gather_output: true,
    });
    assert!(matches!(
        linear.forward(&input, &mut output, &ctx2),
        Err(OpError::Unsupported { .. })
    ));
    let embedding = embedding.with_parallelism(EmbeddingParallelism::Vocab {
        tp,
        vocab_start: 0,
        global_vocab_size: 6,
    });
    assert!(matches!(
        embedding.forward(&ids, &mut hidden, &ctx2),
        Err(OpError::Unsupported { .. })
    ));
    assert_eq!(output.to_host_vec().unwrap(), [123.0; 6]);
    assert_eq!(hidden.stream.to_host_vec().unwrap(), [123.0; 512]);

    let bad_bias = Tensor::<f32, Cpu>::zeros([2], &Cpu).unwrap();
    assert!(
        Linear::from_block_quant(
            BlockQuantProjection::try_new(vec![w]).unwrap(),
            Some(bad_bias)
        )
        .is_err()
    );
}

#[test]
fn dense_embedding_shared_linear_still_compute() {
    let dense = Tensor::from_host_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2], &Cpu).unwrap();
    let embedding = Embed::new(dense.clone());
    let head = embedding.shared_linear().unwrap();
    assert!(Arc::ptr_eq(
        dense.storage(),
        head.weight.as_dense().unwrap().storage()
    ));
    let scope = HostScope::new(Cpu);
    let plan = plan();
    let ctx = StepCtx::new(&scope, &plan);
    let ids = Tensor::from_host_slice(&[1i32, 0], [2], &Cpu).unwrap();
    let mut hidden = Hidden {
        stream: Tensor::<f32, Cpu>::zeros([2, 2], &Cpu).unwrap(),
        pending: None,
    };
    embedding.forward(&ids, &mut hidden, &ctx).unwrap();
    assert_eq!(hidden.stream.to_host_vec().unwrap(), [3.0, 4.0, 1.0, 2.0]);
    let mut logits = Tensor::<f32, Cpu>::zeros([2, 2], &Cpu).unwrap();
    head.forward(&hidden.stream, &mut logits, &ctx).unwrap();
    assert_eq!(logits.to_host_vec().unwrap(), [11.0, 25.0, 5.0, 11.0]);
}

fn fixture_weight(format: BlockQuantFormat, rows: usize) -> (BlockQuantWeight<Cpu>, Vec<f32>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../infer-backend-cpu/tests/fixtures/block_quant/{format:?}.bin"
    ));
    let data = std::fs::read(path).unwrap();
    let b = format.layout();
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for i in 0..rows * (256 / b.elements) {
        let case = 5 + i % 3;
        let offset = 4 + case * (b.bytes + 4 * b.elements);
        bytes.extend_from_slice(&data[offset..offset + b.bytes]);
        expected.extend(
            data[offset + b.bytes..offset + b.bytes + 4 * b.elements]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap())),
        );
    }
    let layout = BlockQuantLayout::new(format, rows, 256).unwrap();
    (
        BlockQuantWeight::from_host(BlockQuantView::new(layout, &bytes).unwrap(), &Cpu).unwrap(),
        expected,
    )
}

#[test]
fn mixed_quantized_linear_executes_multiple_tokens_with_bias_and_strided_output() {
    use BlockQuantFormat::{IQ4_NL, Q4_K, Q5_K};
    let mut expected = vec![];
    let mut weights = vec![];
    for f in [IQ4_NL, Q4_K, Q5_K] {
        let (w, y) = fixture_weight(f, 2);
        weights.push(w);
        expected.extend(y);
    }
    let projection = BlockQuantProjection::try_new(weights).unwrap();
    let bias = Tensor::from_host_slice(&[0.5f32, -0.25, 1.0, 2.0, -2.0, 0.125], [6], &Cpu).unwrap();
    let linear = Linear::from_block_quant(projection, Some(bias.clone())).unwrap();
    let xdata: Vec<f32> = (0..512)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let x = Tensor::from_host_slice(&xdata, [2, 256], &Cpu).unwrap();
    let parent = Tensor::from_host_slice(&[123.0f32; 20], [2, 10], &Cpu).unwrap();
    let mut out = parent.narrow(1, 2, 6).unwrap();
    let scope = HostScope::new(Cpu);
    let plan = plan();
    let ctx = StepCtx::new(&scope, &plan);
    linear.forward(&x, &mut out, &ctx).unwrap();
    let actual = parent.to_host_vec().unwrap();
    let bias = bias.to_host_vec().unwrap();
    for m in 0..2 {
        for n in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..256 {
                sum += xdata[m * 256 + k] * expected[n * 256 + k];
            }
            assert_eq!(actual[m * 10 + 2 + n], sum + bias[n]);
        }
        for n in [0, 1, 8, 9] {
            assert_eq!(actual[m * 10 + n], 123.0);
        }
    }
}

#[test]
fn quantized_embedding_and_tied_head_execute() {
    let (w, expected) = fixture_weight(BlockQuantFormat::Q3_K, 3);
    let embed = Embed::<f32, Cpu>::from_block_quant(w);
    let head = embed.shared_linear().unwrap();
    let scope = HostScope::new(Cpu);
    let plan = plan();
    let ctx = StepCtx::new(&scope, &plan);
    let ids = Tensor::from_host_slice(&[2i32, 0], [2], &Cpu).unwrap();
    let mut hidden = Hidden {
        stream: Tensor::zeros([2, 256], &Cpu).unwrap(),
        pending: None,
    };
    embed.forward(&ids, &mut hidden, &ctx).unwrap();
    let h = hidden.stream.to_host_vec().unwrap();
    assert_eq!(&h[..256], &expected[512..]);
    assert_eq!(&h[256..], &expected[..256]);
    let mut out = Tensor::zeros([2, 3], &Cpu).unwrap();
    head.forward(&hidden.stream, &mut out, &ctx).unwrap();
    let out = out.to_host_vec().unwrap();
    for m in 0..2 {
        for n in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..256 {
                sum += h[m * 256 + k] * expected[n * 256 + k];
            }
            assert_eq!(out[m * 3 + n], sum);
        }
    }
}

#[test]
#[ignore = "set RUSTINFER_GGUF_MODEL to the downloaded Qwen3.8 GGUF"]
fn block_quant_local_cpu_computation() {
    use infer_backend_cpu::block_quant::decode_row;
    let path = std::env::var_os("RUSTINFER_GGUF_MODEL").expect("RUSTINFER_GGUF_MODEL");
    let reader = GgufReader::open(path).unwrap();
    let mut formats = std::collections::HashSet::new();
    let mut matrices = 0;
    let mut values = 0;
    let scope = HostScope::new(Cpu);
    let plan = plan();
    let ctx = StepCtx::new(&scope, &plan);
    for info in reader.tensors() {
        let Some(format) = info.ggml_type().block_quant_format() else {
            continue;
        };
        let view = block_quant_view(reader.read_view(info.name()).unwrap()).unwrap();
        let [n, k] = view.layout().shape();
        let mut row = vec![0.0; k];
        for i in [0, n / 2, n - 1] {
            decode_row(view, i, &mut row).unwrap();
            assert!(row.iter().all(|v| v.is_finite()), "{} row {i}", info.name());
            values += k;
        }
        if formats.insert(format) {
            // Two real rows per encoding; keeps the test small without uploading full weights.
            let sub = view.slice_rows(n - 2..n).unwrap();
            let w = BlockQuantWeight::from_host(sub, &Cpu).unwrap();
            let embed = Embed::<f32, Cpu>::from_block_quant(w);
            let ids = Tensor::from_host_slice(&[1i32, 0], [2], &Cpu).unwrap();
            let mut hidden = Hidden {
                stream: Tensor::zeros([2, k], &Cpu).unwrap(),
                pending: None,
            };
            embed.forward(&ids, &mut hidden, &ctx).unwrap();
            let h = hidden.stream.to_host_vec().unwrap();
            decode_row(sub, 1, &mut row).unwrap();
            assert_eq!(&h[..k], &row);
            decode_row(sub, 0, &mut row).unwrap();
            assert_eq!(&h[k..], &row);
            let xdata: Vec<f32> = (0..2 * k)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.015625)
                .collect();
            let x = Tensor::from_host_slice(&xdata, [2, k], &Cpu).unwrap();
            let mut out = Tensor::zeros([2, 2], &Cpu).unwrap();
            embed
                .shared_linear()
                .unwrap()
                .forward(&x, &mut out, &ctx)
                .unwrap();
            let result = out.to_host_vec().unwrap();
            for m in 0..2 {
                for j in 0..2 {
                    decode_row(sub, j, &mut row).unwrap();
                    let mut sum = 0.0f64;
                    let mut magnitude = 0.0f64;
                    for p in 0..k {
                        let v = xdata[m * k + p] as f64 * row[p] as f64;
                        sum += v;
                        magnitude += v.abs();
                    }
                    // FP32 serial sum compared with FP64 dense dot; norm-scaled tolerance.
                    assert!(
                        (result[m * 2 + j] as f64 - sum).abs() <= 1e-5 * magnitude + 1e-6,
                        "{}",
                        info.name()
                    );
                }
            }
            eprintln!(
                "{format:?}: {} [2,{k}], Embedding + Linear passed",
                info.name()
            );
        }
        matrices += 1;
    }
    assert_eq!(matrices, 506);
    assert_eq!(formats.len(), 13);
    // Actual layer 3 Q/K/V use different encodings. Sample their first two rows
    // and exercise the assembled multi-token Linear, preserving each format.
    let mut parts = vec![];
    let mut dense = vec![];
    for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
        let name = format!("blk.3.{suffix}");
        let view = block_quant_view(reader.read_view(&name).unwrap())
            .unwrap()
            .slice_rows(0..2)
            .unwrap();
        for r in 0..2 {
            let mut row = vec![0.0; 5120];
            decode_row(view, r, &mut row).unwrap();
            dense.extend(row);
        }
        parts.push(BlockQuantWeight::from_host(view, &Cpu).unwrap());
    }
    let linear =
        Linear::from_block_quant(BlockQuantProjection::try_new(parts).unwrap(), None).unwrap();
    let xdata: Vec<f32> = (0..10240)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.015625)
        .collect();
    let x = Tensor::from_host_slice(&xdata, [2, 5120], &Cpu).unwrap();
    let mut out = Tensor::zeros([2, 6], &Cpu).unwrap();
    linear.forward(&x, &mut out, &ctx).unwrap();
    let result = out.to_host_vec().unwrap();
    for m in 0..2 {
        for n in 0..6 {
            let mut sum = 0.0f32;
            for k in 0..5120 {
                sum += xdata[m * 5120 + k] * dense[n * 5120 + k];
            }
            assert_eq!(result[m * 6 + n], sum);
        }
    }
    eprintln!(
        "Decoded {values} finite values from 3 rows of each of {matrices} matrices; real mixed Q/K/V passed"
    );
}

fn check_file(path: &Path) -> usize {
    let reader = GgufReader::open(path).unwrap();
    let mut count = 0;
    for info in reader.tensors() {
        let raw = reader.read_view(info.name()).unwrap();
        if let Some(format) = info.ggml_type().block_quant_format() {
            let view = block_quant_view(raw).unwrap();
            assert_eq!(view.layout().format(), format);
            assert_eq!(
                view.layout().shape(),
                [info.dimensions()[1] as usize, info.dimensions()[0] as usize]
            );
            assert_eq!(view.bytes().len() as u64, info.byte_len());
            let n = view.layout().shape()[0];
            let last = view.slice_rows(n - 1..n).unwrap();
            let range = view.layout().row_byte_range(n - 1..n).unwrap();
            assert_eq!(last.bytes().as_ptr(), view.bytes()[range.clone()].as_ptr());
            // One row only; no whole-model/device upload.
            let copied = BlockQuantWeight::from_host(last, &Cpu).unwrap();
            assert_eq!(copied.bytes().to_host_vec().unwrap(), last.bytes());
            count += 1;
        } else {
            assert!(matches!(
                block_quant_view(raw),
                Err(OpError::Unsupported { .. })
            ));
        }
    }
    count
}

#[test]
fn gguf_fixture_to_block_weights() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf");
    assert_eq!(check_file(&root.join("reference.gguf")), 13);
    let reader = GgufReader::open(root.join("reference.gguf")).unwrap();
    let expected = reader.read_view("test.q3_k").unwrap().bytes.to_vec();
    let owned = BlockQuantWeight::from_host(
        block_quant_view(reader.read_view("test.q3_k").unwrap()).unwrap(),
        &Cpu,
    )
    .unwrap();
    drop(reader);
    assert_eq!(owned.bytes().to_host_vec().unwrap(), expected);
}

#[test]
#[ignore = "set RUSTINFER_GGUF_MODEL to the downloaded Qwen3.8-27B UD-Q3_K_XL file"]
fn block_quant_local_model() {
    let path = std::env::var_os("RUSTINFER_GGUF_MODEL").expect("RUSTINFER_GGUF_MODEL");
    let count = check_file(Path::new(&path));
    assert_eq!(count, 506);
    eprintln!("Validated {count} quantized matrices and copied one complete row from each to CPU");
}

#[path = "block_quant_weights/memory.rs"]
mod memory;

#[cfg(feature = "cute-dsl")]
#[test]
#[ignore = "requires CUDA and RUSTINFER_GGUF_MODEL"]
fn block_quant_local_cuda_embedding() {
    use infer_backend_cpu::block_quant::decode_row;
    use infer_backend_cuda::{Cuda, CudaMemoryPlan};
    use infer_core::exec::ExecScope;
    let reader =
        GgufReader::open(std::env::var_os("RUSTINFER_GGUF_MODEL").expect("RUSTINFER_GGUF_MODEL"))
            .unwrap();
    let view = block_quant_view(reader.read_view("token_embd.weight").unwrap()).unwrap();
    let [n, k] = view.layout().shape();
    assert_eq!([n, k], [248320, 5120]);
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
    assert!(scope.device().config.cute_dsl_available());
    let weight = BlockQuantWeight::from_host(view, scope.device()).unwrap();
    let ids = vec![0i32, 42, 248044, 248046, 248055, (n - 1) as i32, 42];
    let input = Tensor::from_host_slice(&ids, [ids.len()], scope.device()).unwrap();
    let mut p = plan();
    p.num_tokens = ids.len();
    p.batch = ids.len();
    p.q_lens = vec![1; ids.len()];
    p.kv_lens = vec![1; ids.len()];
    p.seq_positions = vec![0; ids.len()];
    p.rope_positions = vec![0; ids.len()];
    let ctx = StepCtx::new(&scope, &p);
    let mut expected = vec![];
    for &id in &ids {
        let mut row = vec![0.0; k];
        decode_row(view, id as usize, &mut row).unwrap();
        expected.extend(row);
    }
    let embed = Embed::<f32, Cuda>::from_block_quant(weight.clone());
    let mut hidden = Hidden {
        stream: Tensor::zeros([ids.len(), k], scope.device()).unwrap(),
        pending: None,
    };
    embed.forward(&input, &mut hidden, &ctx).unwrap();
    scope.synchronize().unwrap();
    assert_eq!(hidden.stream.to_host_vec().unwrap(), expected);
    let embed = Embed::<half::bf16, Cuda>::from_block_quant(weight.clone());
    let mut hidden = Hidden {
        stream: Tensor::zeros([ids.len(), k], scope.device()).unwrap(),
        pending: None,
    };
    embed.forward(&input, &mut hidden, &ctx).unwrap();
    scope.synchronize().unwrap();
    let expected: Vec<half::bf16> = expected.into_iter().map(half::bf16::from_f32).collect();
    assert_eq!(hidden.stream.to_host_vec().unwrap(), expected);
    eprintln!(
        "CuTe GPU Embedding: full Q3_K table [{n},{k}], {} compressed bytes, {} tokens, f32/bf16 match CPU",
        view.bytes().len(),
        ids.len()
    );
}
