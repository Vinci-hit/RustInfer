//! Real two-GPU NCCL integration coverage.
//!
//! Run explicitly on a host with at least two visible CUDA devices:
//! `source scripts/lib/cuda_env.sh && rustinfer_discover_cuda_libraries`
//! `cargo test -p infer-backend-cuda --test nccl -- --ignored --nocapture --test-threads=1`

use std::sync::{Arc, Barrier};
use std::thread;

use half::bf16;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope, NcclCommunicator};
use infer_core::exec::{ExecScope, RankPair, TopologyShape};
use infer_core::ports::{CollectiveOps, CommAxis, ReduceOp};
use infer_core::tensor::Tensor;

const WORLD_SIZE: usize = 2;
const MIB: usize = 1024 * 1024;

#[derive(Debug)]
struct RankResult {
    f32_sum: Vec<f32>,
    bf16_sum: Vec<f32>,
    gathered_rows: Vec<f32>,
}

#[test]
#[ignore = "requires two visible CUDA GPUs and a working NCCL installation"]
fn two_gpu_collectives_preserve_values_and_dim1_layout() {
    let memory_plan = CudaMemoryPlan {
        kernel_workspace_bytes: MIB,
        graph_arena_bytes: 0,
        pool_retain_bytes: MIB,
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
                let topology = TopologyShape {
                    tp: RankPair {
                        rank,
                        size: WORLD_SIZE,
                    },
                    ..TopologyShape::SINGLE
                };
                let scope = CudaScope::new(device)
                    .with_topology(topology)
                    .expect("configure CUDA scope topology")
                    .with_tp_communicator(communicator)
                    .expect("attach TP communicator to CUDA scope");
                assert!(
                    !scope.supports_graphs(),
                    "TP collectives must stay eager until rank-synchronized graph capture exists"
                );
                let device = scope.device().clone();
                // Tensor allocation/upload and teardown must happen with this
                // rank's CUDA device current on the worker thread.
                let _active_device = scope.enter();

                let f32_input = if rank == 0 {
                    vec![1.0f32, 2.0, 3.0, 4.0]
                } else {
                    vec![10.0f32, 20.0, 30.0, 40.0]
                };
                let mut f32_tensor =
                    Tensor::from_host_slice(&f32_input, [2, 2], &device).expect("upload f32 shard");
                rendezvous.wait();
                <Cuda as CollectiveOps>::all_reduce(
                    &scope,
                    CommAxis::Tp,
                    ReduceOp::Sum,
                    &mut f32_tensor,
                )
                .expect("f32 NCCL all-reduce");
                scope.synchronize().expect("synchronize f32 all-reduce");
                let f32_sum = f32_tensor.to_host_vec().expect("download f32 sum");

                let bf16_input_f32 = if rank == 0 {
                    [1.0f32, 2.0, 4.0, 8.0]
                } else {
                    [16.0f32, 32.0, 64.0, 128.0]
                };
                let bf16_input: Vec<bf16> =
                    bf16_input_f32.into_iter().map(bf16::from_f32).collect();
                let mut bf16_tensor = Tensor::from_host_slice(&bf16_input, [2, 2], &device)
                    .expect("upload bf16 shard");
                rendezvous.wait();
                <Cuda as CollectiveOps>::all_reduce(
                    &scope,
                    CommAxis::Tp,
                    ReduceOp::Sum,
                    &mut bf16_tensor,
                )
                .expect("bf16 NCCL all-reduce");
                scope.synchronize().expect("synchronize bf16 all-reduce");
                let bf16_sum = bf16_tensor
                    .to_host_vec()
                    .expect("download bf16 sum")
                    .into_iter()
                    .map(|value| value.to_f32())
                    .collect();

                // Each rank owns two columns from two rows. Gathering dim=1
                // must concatenate rank shards within each row, rather than
                // flattening all rows rank-by-rank.
                let row_shard = if rank == 0 {
                    vec![1.0f32, 2.0, 3.0, 4.0]
                } else {
                    vec![10.0f32, 20.0, 30.0, 40.0]
                };
                let row_shard =
                    Tensor::from_host_slice(&row_shard, [2, 2], &device).expect("upload row shard");
                let mut gathered =
                    Tensor::<f32, Cuda>::zeros([2, 4], &device).expect("allocate gather output");
                rendezvous.wait();
                <Cuda as CollectiveOps>::all_gather(
                    &scope,
                    CommAxis::Tp,
                    1,
                    &row_shard,
                    &mut gathered,
                )
                .expect("dim=1 NCCL all-gather");
                scope.synchronize().expect("synchronize all-gather");
                let gathered_rows = gathered.to_host_vec().expect("download gathered rows");

                RankResult {
                    f32_sum,
                    bf16_sum,
                    gathered_rows,
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
                .unwrap_or_else(|_| panic!("NCCL rank thread {rank} panicked"))
        })
        .collect();

    for (rank, result) in results.iter().enumerate() {
        assert_eq!(
            result.f32_sum,
            [11.0, 22.0, 33.0, 44.0],
            "rank {rank} f32 all-reduce"
        );
        assert_eq!(
            result.bf16_sum,
            [17.0, 34.0, 68.0, 136.0],
            "rank {rank} bf16 all-reduce"
        );
        assert_eq!(
            result.gathered_rows,
            [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
            "rank {rank} dim=1 all-gather"
        );
    }
}
