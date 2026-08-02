#![cfg(feature = "cuda")]

//! Real two-GPU coverage for tensor-parallel worker components.
//!
//! Run explicitly on a host with at least two visible CUDA devices:
//! `source scripts/lib/cuda_env.sh && rustinfer_discover_cuda_libraries`
//! `cargo test -p infer-worker --test tp_components -- --ignored --nocapture --test-threads=1`

use std::sync::{Arc, Barrier};
use std::thread;

use half::bf16;
use infer_worker::components::embed::{Embed, EmbeddingParallelism};
use infer_worker::components::linear::{Linear, LinearParallelism};
use infer_worker::domain::component::Hidden;
use infer_worker::domain::exec::{ExecScope, RankPair, StepCtx, TopologyShape};
use infer_worker::domain::plan::{BatchKind, BatchPlan};
use infer_worker::domain::tensor::Tensor;
use infer_worker::infrastructure::cuda::{Cuda, CudaMemoryPlan, CudaScope, NcclCommunicator};

const WORLD_SIZE: usize = 2;
const MIB: usize = 1024 * 1024;

#[derive(Debug)]
struct RankResult {
    replicated_embedding: Vec<f32>,
    embedding: Vec<f32>,
    vocab_logits: Vec<f32>,
    row_output: Vec<f32>,
}

fn bf16_array<const N: usize>(values: [f32; N]) -> [bf16; N] {
    values.map(bf16::from_f32)
}

fn batch_plan(num_tokens: usize) -> BatchPlan {
    BatchPlan {
        kind: BatchKind::DecodeOnly,
        num_tokens,
        batch: num_tokens,
        q_lens: vec![1; num_tokens],
        kv_lens: vec![1; num_tokens],
        seq_positions: vec![0; num_tokens],
        rope_positions: vec![0; num_tokens],
        max_blocks_per_seq: 1,
        block_size: 1,
        total_q_tiles: 0,
    }
}

#[test]
#[ignore = "requires two visible CUDA GPUs and a working NCCL installation"]
fn two_gpu_tp_components_reconstruct_global_results() {
    let memory_plan = CudaMemoryPlan {
        kernel_workspace_bytes: 8 * MIB,
        graph_arena_bytes: 0,
        pool_retain_bytes: 8 * MIB,
    };
    let devices: Vec<Cuda> = (0..WORLD_SIZE)
        .map(|device_id| {
            Cuda::with_memory_plan(device_id as i32, memory_plan)
                .unwrap_or_else(|error| panic!("create CUDA device {device_id}: {error:?}"))
        })
        .collect();
    let communicators =
        NcclCommunicator::init_all(&devices).expect("initialize two-rank NCCL communicator");
    let rendezvous = Arc::new(Barrier::new(WORLD_SIZE));

    let handles: Vec<_> = devices
        .into_iter()
        .zip(communicators)
        .enumerate()
        .map(|(rank, (device, communicator))| {
            let rendezvous = Arc::clone(&rendezvous);
            thread::spawn(move || {
                let tp = RankPair {
                    rank,
                    size: WORLD_SIZE,
                };
                let scope = CudaScope::new(device)
                    .with_topology(TopologyShape {
                        tp,
                        ..TopologyShape::SINGLE
                    })
                    .expect("configure CUDA scope topology")
                    .with_tp_communicator(communicator)
                    .expect("attach TP communicator to CUDA scope");
                let device = scope.device().clone();
                // Allocations, uploads, downloads, and destruction all happen
                // while this rank's CUDA device is current on its thread.
                let _active_device = scope.enter();

                // Replicated embedding is the vocab_start=0 specialization of
                // the same kernel. An 8-wide BF16 row exercises its float4
                // vectorized path independently on both ranks.
                let replicated_table: Vec<bf16> = (1..=4)
                    .flat_map(|value| std::iter::repeat_n(bf16::from_f32(value as f32), 8))
                    .collect();
                let replicated_embed = Embed::new(
                    Tensor::from_host_slice(&replicated_table, [4, 8], &device)
                        .expect("upload replicated embedding table"),
                )
                .with_parallelism(EmbeddingParallelism::Replicated { tp });
                let replicated_ids = Tensor::from_host_slice(&[3_i32, 0], [2], &device)
                    .expect("upload replicated embedding token ids");
                let mut replicated_hidden = Hidden {
                    stream: Tensor::<bf16, Cuda>::zeros([2, 8], &device)
                        .expect("allocate replicated embedding output"),
                    pending: None,
                };
                let replicated_plan = batch_plan(2);
                let replicated_ctx = StepCtx::new(&scope, &replicated_plan);
                replicated_embed
                    .forward(&replicated_ids, &mut replicated_hidden, &replicated_ctx)
                    .expect("replicated embedding forward");
                scope
                    .synchronize()
                    .expect("synchronize replicated embedding");
                let replicated_embedding = replicated_hidden
                    .stream
                    .to_host_vec()
                    .expect("download replicated embedding")
                    .into_iter()
                    .map(|value| value.to_f32())
                    .collect();

                // Rank 0 owns tokens [0, 2), rank 1 owns [2, 4). Each local
                // lookup masks the other rank's tokens; the all-reduce restores
                // the complete embedding for a mixed-rank token sequence.
                let local_embedding = if rank == 0 {
                    bf16_array([1.0, 10.0, 100.0, 2.0, 20.0, 200.0])
                } else {
                    bf16_array([3.0, 30.0, 300.0, 4.0, 40.0, 400.0])
                };
                let table = Tensor::from_host_slice(&local_embedding, [2, 3], &device)
                    .expect("upload local embedding shard");
                let input_ids = Tensor::from_host_slice(&[0_i32, 3, 2, 1], [4], &device)
                    .expect("upload global token ids");
                let mut hidden = Hidden {
                    stream: Tensor::<bf16, Cuda>::zeros([4, 3], &device)
                        .expect("allocate embedding output"),
                    pending: None,
                };
                let embed = Embed::new(table).with_parallelism(EmbeddingParallelism::Vocab {
                    tp,
                    vocab_start: rank * 2,
                    global_vocab_size: 4,
                });
                let embed_plan = batch_plan(4);
                let embed_ctx = StepCtx::new(&scope, &embed_plan);
                rendezvous.wait();
                embed
                    .forward(&input_ids, &mut hidden, &embed_ctx)
                    .expect("vocab-parallel embedding forward");
                scope
                    .synchronize()
                    .expect("synchronize vocab-parallel embedding");
                let embedding = hidden
                    .stream
                    .to_host_vec()
                    .expect("download reconstructed embedding")
                    .into_iter()
                    .map(|value| value.to_f32())
                    .collect();

                // Two local vocabulary columns from each rank are gathered on
                // dim 1. The expected layout interleaves neither ranks nor rows.
                let vocab_weight = if rank == 0 {
                    bf16_array([1.0, 0.0, 0.0, 1.0])
                } else {
                    bf16_array([1.0, 1.0, 2.0, -1.0])
                };
                let vocab_linear = Linear::new(
                    Tensor::from_host_slice(&vocab_weight, [2, 2], &device)
                        .expect("upload local vocabulary weight"),
                    None,
                )
                .with_parallelism(LinearParallelism::Vocab {
                    tp,
                    global_out_features: 4,
                });
                let vocab_input =
                    Tensor::from_host_slice(&bf16_array([1.0, 2.0, 3.0, 4.0]), [2, 2], &device)
                        .expect("upload vocabulary-linear input");
                let mut vocab_output = Tensor::<bf16, Cuda>::zeros([2, 4], &device)
                    .expect("allocate full vocabulary logits");
                let linear_plan = batch_plan(2);
                let linear_ctx = StepCtx::new(&scope, &linear_plan);
                rendezvous.wait();
                vocab_linear
                    .forward(&vocab_input, &mut vocab_output, &linear_ctx)
                    .expect("vocab-parallel linear forward");
                scope
                    .synchronize()
                    .expect("synchronize vocabulary all-gather");
                let vocab_logits = vocab_output
                    .to_host_vec()
                    .expect("download full vocabulary logits")
                    .into_iter()
                    .map(|value| value.to_f32())
                    .collect();

                // Input-feature and weight shards produce partial matmuls. Bias
                // is replicated, so it must be added after the all-reduce and
                // therefore exactly once on every rank.
                let (row_input, row_weight) = if rank == 0 {
                    (
                        bf16_array([1.0, 1.0, 1.0, 2.0]),
                        bf16_array([1.0, 2.0, 5.0, 6.0]),
                    )
                } else {
                    (
                        bf16_array([1.0, 1.0, 3.0, 4.0]),
                        bf16_array([3.0, 4.0, 7.0, 8.0]),
                    )
                };
                let row_linear = Linear::new(
                    Tensor::from_host_slice(&row_weight, [2, 2], &device)
                        .expect("upload row-parallel weight shard"),
                    Some(
                        Tensor::from_host_slice(&bf16_array([0.5, -1.0]), [2], &device)
                            .expect("upload replicated row-parallel bias"),
                    ),
                )
                .with_parallelism(LinearParallelism::Row { tp });
                let row_input = Tensor::from_host_slice(&row_input, [2, 2], &device)
                    .expect("upload row-parallel input shard");
                let mut row_output = Tensor::<bf16, Cuda>::zeros([2, 2], &device)
                    .expect("allocate row-parallel output");
                rendezvous.wait();
                row_linear
                    .forward(&row_input, &mut row_output, &linear_ctx)
                    .expect("row-parallel linear forward");
                scope
                    .synchronize()
                    .expect("synchronize row-parallel all-reduce and bias");
                let row_output = row_output
                    .to_host_vec()
                    .expect("download row-parallel output")
                    .into_iter()
                    .map(|value| value.to_f32())
                    .collect();
                rendezvous.wait();

                RankResult {
                    replicated_embedding,
                    embedding,
                    vocab_logits,
                    row_output,
                }
            })
        })
        .collect();

    let results: Vec<RankResult> = handles
        .into_iter()
        .enumerate()
        .map(|(rank, handle)| {
            handle
                .join()
                .unwrap_or_else(|_| panic!("TP rank thread {rank} panicked"))
        })
        .collect();

    for (rank, result) in results.iter().enumerate() {
        assert_eq!(
            result.replicated_embedding,
            [
                4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ],
            "rank {rank} replicated embedding"
        );
        assert_eq!(
            result.embedding,
            [
                1.0, 10.0, 100.0, 4.0, 40.0, 400.0, 3.0, 30.0, 300.0, 2.0, 20.0, 200.0,
            ],
            "rank {rank} vocab-parallel embedding"
        );
        assert_eq!(
            result.vocab_logits,
            [1.0, 2.0, 3.0, 0.0, 3.0, 4.0, 7.0, 2.0],
            "rank {rank} vocab-parallel logits layout"
        );
        assert_eq!(
            result.row_output,
            [10.5, 25.0, 30.5, 69.0],
            "rank {rank} row-parallel linear must add bias exactly once"
        );
    }
}
