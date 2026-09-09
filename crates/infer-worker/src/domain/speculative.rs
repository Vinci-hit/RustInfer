//! Contracts shared by speculative proposers, verifiers, and state committers.

use std::ops::Range;

use crate::domain::ports::{OpError, OpResult, SampledToken};

/// A verification decision; it does not commit or mutate model state.
///
/// The target consumes `[pending, draft_1, ..., draft_K]`. If `r` drafts
/// are accepted, the committer retains `1 + r` input positions, emits the
/// accepted drafts followed by `correction_or_bonus`, and leaves that last
/// token pending for the next target step.
#[derive(Debug, Clone)]
pub struct Verification {
    pub accepted_drafts: usize,
    pub correction_or_bonus: SampledToken,
}

/// A validated, borrowed ragged batch: K drafts require K + 1 target rows.
/// An empty draft slice for an individual sequence represents K = 0.
#[derive(Debug, Clone, Copy)]
pub struct DraftBatch<'a> {
    drafts: &'a [Vec<i32>],
    q_lens: &'a [i32],
    target_rows: usize,
}

#[derive(Debug)]
pub struct DraftSequence<'a> {
    pub drafts: &'a [i32],
    pub target_rows: Range<usize>,
}

impl<'a> DraftBatch<'a> {
    pub fn new(drafts: &'a [Vec<i32>], q_lens: &'a [i32], target_rows: usize) -> OpResult<Self> {
        if drafts.is_empty() || drafts.len() != q_lens.len() {
            return Err(OpError::Shape(format!(
                "DraftBatch: {} draft sequences require the same nonzero number of q_lens, got {}",
                drafts.len(),
                q_lens.len()
            )));
        }
        let mut rows = 0usize;
        for (seq, (draft, &q_len)) in drafts.iter().zip(q_lens).enumerate() {
            let required = draft
                .len()
                .checked_add(1)
                .ok_or_else(|| OpError::Shape("DraftBatch: draft length overflow".into()))?;
            if q_len <= 0 || q_len as usize != required {
                return Err(OpError::Shape(format!(
                    "DraftBatch: sequence {seq} has {} drafts; requires {required} target rows, got {q_len}",
                    draft.len()
                )));
            }
            rows = rows
                .checked_add(required)
                .ok_or_else(|| OpError::Shape("DraftBatch: target row offset overflow".into()))?;
        }
        if rows != target_rows {
            return Err(OpError::Shape(format!(
                "DraftBatch: requires {rows} target rows, got {target_rows}"
            )));
        }
        Ok(Self {
            drafts,
            q_lens,
            target_rows,
        })
    }

    pub fn batch_size(self) -> usize {
        self.drafts.len()
    }

    pub fn target_rows(self) -> usize {
        self.target_rows
    }

    /// The checked constructor guarantees non-overlapping, in-bounds ranges.
    pub fn sequences(self) -> impl Iterator<Item = DraftSequence<'a>> {
        self.drafts
            .iter()
            .zip(self.q_lens)
            .scan(0usize, |offset, (drafts, &q_len)| {
                let start = *offset;
                *offset += q_len as usize;
                Some(DraftSequence {
                    drafts,
                    target_rows: start..*offset,
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_ranges_include_one_extra_row_for_each_sequence() {
        let drafts = vec![vec![1, 2], vec![], vec![3]];
        let batch = DraftBatch::new(&drafts, &[3, 1, 2], 6).unwrap();
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.target_rows(), 6);
        assert_eq!(
            batch
                .sequences()
                .map(|seq| seq.target_rows)
                .collect::<Vec<_>>(),
            vec![0..3, 3..4, 4..6]
        );
    }

    #[test]
    fn malformed_layouts_fail_before_any_prediction_access() {
        assert!(DraftBatch::new(&[], &[], 0).is_err());
        assert!(DraftBatch::new(&[vec![]], &[], 1).is_err());
        assert!(DraftBatch::new(&[vec![]], &[0], 0).is_err());
        assert!(DraftBatch::new(&[vec![]], &[-1], 0).is_err());
        assert!(DraftBatch::new(&[vec![1]], &[1], 1).is_err());
        assert!(DraftBatch::new(&[vec![1]], &[3], 3).is_err());
        assert!(DraftBatch::new(&[vec![1]], &[2], 1).is_err());
        assert!(DraftBatch::new(&[vec![1]], &[2], 3).is_err());
    }
}
