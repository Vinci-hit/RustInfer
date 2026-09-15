use infer_worker::domain::{
    kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool},
    plan::{BatchKind, BatchPlan},
    tensor::Tensor,
};

pub(super) fn indices<D: infer_worker::domain::ports::backend::LlmBackend>(
    start: usize,
    n: usize,
    blocks: usize,
    device: &D,
) -> (BatchPlan, KvIndexTensors<D>) {
    let positions: Vec<i32> = (start as i32..(start + n) as i32).collect();
    let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&[n as i32]);
    let plan = BatchPlan {
        kind: if n == 1 {
            BatchKind::DecodeOnly
        } else {
            BatchKind::Ragged
        },
        num_tokens: n,
        batch: 1,
        q_lens: vec![n as i32],
        kv_lens: vec![(start + n) as i32],
        seq_positions: vec![start as i32],
        rope_positions: positions.clone(),
        max_blocks_per_seq: blocks,
        block_size: 1,
        total_q_tiles: req.len() as i32,
    };
    let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], device).unwrap();
    let idx = KvIndexTensors {
        decode_rows: None,
        block_tables: Tensor::from_host_slice(
            &(0..blocks as i32).collect::<Vec<_>>(),
            [1, blocks],
            device,
        )
        .unwrap(),
        cu_q_lens: ints(&cu),
        kv_lens: ints(&plan.kv_lens),
        seq_positions: ints(&[start as i32]),
        seq_lens_step: ints(&[n as i32]),
        rope_positions: ints(&positions),
        block2req: ints(&req),
        block2tile: ints(&tile),
        valid_q_tiles: ints(&[req.len() as i32]),
        valid_suffix_q_tiles: ints(&[req.len() as i32]),
    };
    (plan, idx)
}
pub(super) fn pool<
    T: infer_worker::domain::dtype::Dtype,
    D: infer_worker::domain::ports::backend::LlmBackend,
>(
    layers: usize,
    blocks: usize,
    kv_dim: usize,
    device: &D,
) -> PagedKvPool<T, D> {
    PagedKvPool {
        layers: (0..layers)
            .map(|_| PagedKvLayer {
                k: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
                v: Tensor::zeros([blocks, 1, kv_dim], device).unwrap(),
            })
            .collect(),
        num_blocks: blocks,
        block_size: 1,
        kv_dim,
        quant: KvQuantTier::None,
        seq_kv_len: Default::default(),
    }
}
