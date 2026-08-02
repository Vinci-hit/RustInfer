use infer_core::dtype::Dtype;
use infer_core::exec::ExecDevice as Device;
use infer_core::tensor::Tensor;

use crate::ports::OpResult;

/// Backend operations specific to vocabulary-parallel model boundaries.
pub trait VocabOps: Device {
    /// Lookup global token ids in a vocabulary-row shard. Tokens outside this
    /// rank's `[vocab_start, vocab_start + local_rows)` range write zeros; the
    /// caller combines rank-local outputs with an AllReduce(Sum).
    fn vocab_embedding<T: Dtype>(
        scope: &Self::Scope,
        table: &Tensor<T, Self>,
        global_indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
        vocab_start: usize,
        global_vocab_size: usize,
    ) -> OpResult<()>;
}
