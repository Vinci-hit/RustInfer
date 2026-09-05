#![cfg(feature = "cuda")]

//! Opt-in GPU differential coverage for the complete GDN component, including
//! its projections, shared scratch, ragged prefill and reordered decode slots.

use half::bf16;
use infer_worker::components::{GatedDeltaNet, GdnWeights, Linear, RmsNorm};
use infer_worker::domain::cache::{LinearBatch, LinearDims, LinearLayerState};
use infer_worker::domain::component::Hidden;
use infer_worker::domain::exec::{ExecScope, HostScope, StepCtx};
use infer_worker::domain::gdn_scratch::GdnScratch;
use infer_worker::domain::plan::{BatchKind, BatchPlan};
use infer_worker::domain::ports::backend::LlmBackend;
use infer_worker::domain::tensor::Tensor;
use infer_worker::infrastructure::cpu::Cpu;
use infer_worker::infrastructure::cuda::{Cuda, CudaMemoryPlan, CudaScope};

const DIM: usize = 32;
const DIMS: LinearDims = LinearDims {
    num_key_heads: 2,
    num_value_heads: 4,
    key_head_dim: 16,
    value_head_dim: 16,
    conv_kernel_dim: 4,
};

fn tensor<D: LlmBackend>(device: &D, rows: usize, cols: usize, phase: f32) -> Tensor<bf16, D> {
    let values: Vec<bf16> = (0..rows * cols)
        .map(|i| bf16::from_f32(((i as f32 * 0.037 + phase).sin() + 0.1) * 0.12))
        .collect();
    Tensor::from_host_slice(&values, [rows, cols], device).unwrap()
}

fn run<D: LlmBackend>(scope: &D::Scope) -> Vec<Vec<f32>> {
    let _guard = scope.enter();
    let device = scope.device();
    let linear = |rows, cols, phase| Linear::new(tensor(device, rows, cols, phase), None);
    let mut model = GatedDeltaNet::new(
        GdnWeights {
            input_layernorm: RmsNorm {
                weight: Tensor::from_host_slice(&[bf16::from_f32(1.0); DIM], [DIM], device)
                    .unwrap(),
                eps: 0.0,
            },
            in_proj_qkv: linear(DIMS.conv_dim(), DIM, 0.1),
            in_proj_a: linear(DIMS.num_value_heads, DIM, 0.7),
            in_proj_b: linear(DIMS.num_value_heads, DIM, 1.3),
            in_proj_z: linear(DIMS.value_dim(), DIM, 1.9),
            conv1d: tensor(device, DIMS.conv_dim(), 4, 2.1)
                .view_contiguous([DIMS.conv_dim(), 1, 4].into())
                .unwrap(),
            a_log: Tensor::from_host_slice(&[-0.3_f32, 0.1, -0.1, 0.2], [4], device).unwrap(),
            dt_bias: Tensor::from_host_slice(&[bf16::from_f32(0.1); 4], [4], device).unwrap(),
            norm_weight: Tensor::from_host_slice(
                &(0..16).map(|i| 0.7 + i as f32 * 0.03).collect::<Vec<_>>(),
                [16],
                device,
            )
            .unwrap(),
            norm_eps: 1e-6,
            out_proj: linear(DIM, DIMS.value_dim(), 2.7),
        },
        DIMS,
    )
    .unwrap();
    model
        .install_scratch(GdnScratch::new(device, DIM, DIMS, 8).unwrap())
        .unwrap();
    let mut state = LinearLayerState::new(DIMS, 3, device).unwrap();
    let mut results = Vec::new();
    for (step, (slots, q_lens)) in [([2, 0], [3, 1]), ([0, 2], [1, 1])].into_iter().enumerate() {
        let num_tokens = q_lens.iter().sum::<i32>() as usize;
        let batch = LinearBatch::new(&slots, &q_lens, 3, device).unwrap();
        let plan = BatchPlan {
            kind: if step == 0 {
                BatchKind::Ragged
            } else {
                BatchKind::DecodeOnly
            },
            num_tokens,
            batch: 2,
            q_lens: q_lens.to_vec(),
            kv_lens: vec![4, 4],
            seq_positions: vec![0, 0],
            rope_positions: vec![0; num_tokens],
            max_blocks_per_seq: 4,
            block_size: 1,
            total_q_tiles: 2,
        };
        let ctx = StepCtx::new(scope, &plan);
        // CUDA's existing BF16 input norm rounds inv_rms before multiplying;
        // the CPU fallback only rounds the final result. Use exact ±0.25
        // residual sums (inv_rms=4) to align this boundary while exercising
        // nonzero pending deltas and different token/column sign patterns.
        let activation = |scale: f32| {
            let values: Vec<bf16> = (0..num_tokens * DIM)
                .map(|i| {
                    let sign: f32 = if (i * 7 + i / DIM + step) % 11 < 5 {
                        -1.0
                    } else {
                        1.0
                    };
                    bf16::from_f32(sign * scale)
                })
                .collect();
            Tensor::from_host_slice(&values, [num_tokens, DIM], device).unwrap()
        };
        let mut hidden = Hidden {
            stream: activation(0.1875),
            pending: Some(activation(0.0625)),
        };
        model.run(&mut hidden, state.view(&batch), &ctx).unwrap();
        scope.synchronize().unwrap();
        results.push(
            hidden
                .pending
                .unwrap()
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(f32::from)
                .collect(),
        );
        results.push(
            hidden
                .stream
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(f32::from)
                .collect(),
        );
    }
    results.push(
        state
            .conv()
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(f32::from)
            .collect(),
    );
    results.push(state.ssm().to_host_vec().unwrap());
    results
}

#[test]
#[ignore = "requires a visible CUDA GPU; run with --ignored --test-threads=1"]
fn bf16_gdn_component_matches_cpu_across_prefill_and_decode() {
    let expected = run::<Cpu>(&HostScope::new(Cpu));
    let cuda = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 8 * 1024 * 1024,
            graph_arena_bytes: 0,
            pool_retain_bytes: 8 * 1024 * 1024,
        },
    )
    .unwrap();
    let actual = run::<Cuda>(&CudaScope::new(cuda));
    for (part, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= 0.002 + b.abs() * 0.025,
                "part {part}, index {i}: CUDA={a} CPU={b}"
            );
        }
    }
}
