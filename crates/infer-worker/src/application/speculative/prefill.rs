//! Per-sequence text-prefill alignment for a hidden-conditioned draft head.
//! Only the final normalized hidden row survives a completed chunk.
use crate::domain::dtype::Dtype;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

pub struct MtpPrefill<T: Dtype, D: LlmBackend> {
    pending: Option<(i32, Tensor<T, D>)>,
}
impl<T: Dtype, D: LlmBackend> Default for MtpPrefill<T, D> {
    fn default() -> Self {
        Self { pending: None }
    }
}

/// Dropping an uncommitted chunk preserves the previous alignment state.
/// Commit only after the corresponding head execution succeeds. KV recovery
/// after an execution error is the caller's responsibility.
pub struct PreparedMtpChunk<'a, T: Dtype, D: LlmBackend> {
    state: &'a mut MtpPrefill<T, D>,
    next: (i32, Tensor<T, D>),
    pub next_token_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub target_hidden: Tensor<T, D>,
}
impl<T: Dtype, D: LlmBackend> PreparedMtpChunk<'_, T, D> {
    pub fn commit(self) {
        self.state.pending = Some(self.next);
    }
}

impl<T: Dtype, D: LlmBackend> MtpPrefill<T, D> {
    pub fn pending(&self) -> Option<(i32, &Tensor<T, D>)> {
        self.pending.as_ref().map(|(p, h)| (*p, h))
    }
    pub fn reset(&mut self) {
        self.pending = None;
    }

    /// Hidden rows must be final-normalized target outputs for these tokens.
    /// A first singleton chunk produces zero pairs and must skip head execution.
    /// Copies run on the tensor device's default stream. The caller must make
    /// target outputs ready on that stream before preparing the chunk.
    pub fn prepare(
        &mut self,
        ids: &[i32],
        positions: &[i32],
        hidden: &Tensor<T, D>,
    ) -> OpResult<PreparedMtpChunk<'_, T, D>> {
        let n = ids.len();
        let shape = hidden.shape().as_slice();
        if n == 0
            || positions.len() != n
            || shape.len() != 2
            || shape[0] != n
            || shape[1] == 0
            || !hidden.is_contiguous()
            || positions[0] < 0
            || positions
                .windows(2)
                .any(|p| p[0].checked_add(1) != Some(p[1]))
        {
            return Err(OpError::Shape(
                "MTP prefill requires nonempty consecutive text positions and [tokens, dim] hidden"
                    .into(),
            ));
        }
        if let Some((p, h)) = &self.pending {
            if p.checked_add(1) != Some(positions[0]) || h.shape().as_slice() != [1, shape[1]] {
                return Err(OpError::Shape(
                    "MTP prefill chunk does not follow pending hidden".into(),
                ));
            }
        } else if positions[0] != 0 {
            return Err(OpError::Shape(
                "MTP prefill must start at position zero".into(),
            ));
        }
        let carry = usize::from(self.pending.is_some());
        let count = n - 1 + carry;
        let aligned = Tensor::zeros([count, shape[1]], hidden.device())?;
        let mut next_ids = Vec::with_capacity(count);
        let mut pair_positions = Vec::with_capacity(count);
        if let Some((p, h)) = &self.pending {
            aligned.narrow(0, 0, 1)?.copy_from(h)?;
            next_ids.push(ids[0]);
            pair_positions.push(*p);
        }
        if n > 1 {
            aligned
                .narrow(0, carry, n - 1)?
                .copy_from(&hidden.narrow(0, 0, n - 1)?)?;
            next_ids.extend_from_slice(&ids[1..]);
            pair_positions.extend_from_slice(&positions[..n - 1]);
        }
        let mut last = Tensor::zeros([1, shape[1]], hidden.device())?;
        last.copy_from(&hidden.narrow(0, n - 1, 1)?)?;
        Ok(PreparedMtpChunk {
            state: self,
            next: (positions[n - 1], last),
            next_token_ids: next_ids,
            positions: pair_positions,
            target_hidden: aligned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;
    #[test]
    fn every_chunk_partition_preserves_shift_and_hidden_positions() {
        let ids = [10, 11, 12, 13, 14];
        let all =
            Tensor::from_host_slice(&(0..15).map(|x| x as f32).collect::<Vec<_>>(), [5, 3], &Cpu)
                .unwrap();
        for cuts in 0..16 {
            let mut state = MtpPrefill::default();
            let (mut tokens, mut pos, mut hidden) = (Vec::new(), Vec::new(), Vec::new());
            let mut start = 0;
            for end in 1..=5 {
                if end < 5 && cuts & (1 << (end - 1)) == 0 {
                    continue;
                }
                let ps: Vec<_> = (start as i32..end as i32).collect();
                let chunk = state
                    .prepare(
                        &ids[start..end],
                        &ps,
                        &all.narrow(0, start, end - start).unwrap(),
                    )
                    .unwrap();
                tokens.extend_from_slice(&chunk.next_token_ids);
                pos.extend_from_slice(&chunk.positions);
                hidden.extend(chunk.target_hidden.to_host_vec().unwrap());
                chunk.commit();
                start = end;
            }
            assert_eq!(tokens, ids[1..]);
            assert_eq!(pos, [0, 1, 2, 3]);
            assert_eq!(hidden, (0..12).map(|x| x as f32).collect::<Vec<_>>());
            let (p, h) = state.pending().unwrap();
            assert_eq!(p, 4);
            assert_eq!(h.to_host_vec().unwrap(), [12., 13., 14.]);
        }
    }
    #[test]
    fn failed_chunk_preserves_owned_carry_and_can_retry() {
        let mut state = MtpPrefill::default();
        let mut h = Tensor::from_host_slice(&[1f32, 2.], [1, 2], &Cpu).unwrap();
        state.prepare(&[10], &[0], &h).unwrap().commit();
        h.copy_from(&Tensor::from_host_slice(&[8f32, 9.], [1, 2], &Cpu).unwrap())
            .unwrap();
        assert_eq!(state.pending().unwrap().1.to_host_vec().unwrap(), [1., 2.]);
        drop(state.prepare(&[11], &[1], &h).unwrap());
        assert_eq!(state.pending().unwrap().0, 0);
        assert!(state.prepare(&[12], &[2], &h).is_err());
        state.prepare(&[11], &[1], &h).unwrap().commit();
        assert_eq!(state.pending().unwrap().0, 1);
        state.reset();
        assert!(state.pending().is_none());
    }
}
