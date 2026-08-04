//! Process-local placement within a global tensor-parallel group.
//!
//! Execution topology answers "which global rank is this runtime?".  This
//! type answers the deployment question one level above it: "which contiguous
//! global ranks are owned by this worker process?".  Keeping those concepts
//! separate lets the current single-process worker use all TP ranks locally
//! while leaving a clean boundary for a future one-process-per-node layout.

use std::ops::Range;

use infer_core::ports::{OpError, OpResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelPlacement {
    global_size: usize,
    local_rank_start: usize,
    local_rank_count: usize,
}

impl TensorParallelPlacement {
    /// All ranks in the TP group are owned by the current process.
    pub fn single_process(global_size: usize) -> OpResult<Self> {
        Self::new(global_size, 0, global_size)
    }

    /// Describe a contiguous process-local slice of the global TP ranks.
    pub fn new(
        global_size: usize,
        local_rank_start: usize,
        local_rank_count: usize,
    ) -> OpResult<Self> {
        if global_size == 0 {
            return Err(OpError::Shape(
                "tensor-parallel global size must be greater than zero".into(),
            ));
        }
        if local_rank_count == 0 {
            return Err(OpError::Shape(
                "a tensor-parallel worker must own at least one local rank".into(),
            ));
        }
        let local_rank_end = local_rank_start
            .checked_add(local_rank_count)
            .ok_or_else(|| OpError::Shape("tensor-parallel local rank range overflowed".into()))?;
        if local_rank_end > global_size {
            return Err(OpError::Shape(format!(
                "tensor-parallel local rank range {local_rank_start}..{local_rank_end} exceeds global size {global_size}"
            )));
        }
        Ok(Self {
            global_size,
            local_rank_start,
            local_rank_count,
        })
    }

    pub const fn global_size(self) -> usize {
        self.global_size
    }

    pub const fn local_rank_start(self) -> usize {
        self.local_rank_start
    }

    pub const fn local_rank_count(self) -> usize {
        self.local_rank_count
    }

    pub fn owned_global_ranks(self) -> Range<usize> {
        self.local_rank_start..self.local_rank_start + self.local_rank_count
    }

    pub const fn owns_global_root(self) -> bool {
        self.local_rank_start == 0
    }

    pub const fn is_single_process(self) -> bool {
        self.local_rank_start == 0 && self.local_rank_count == self.global_size
    }

    pub fn global_rank(self, local_rank: usize) -> OpResult<usize> {
        if local_rank >= self.local_rank_count {
            return Err(OpError::Shape(format!(
                "local TP rank {local_rank} is outside process-local size {}",
                self.local_rank_count
            )));
        }
        Ok(self.local_rank_start + local_rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_process_owns_the_complete_global_group() {
        let placement = TensorParallelPlacement::single_process(4).unwrap();

        assert_eq!(placement.global_size(), 4);
        assert_eq!(placement.local_rank_start(), 0);
        assert_eq!(placement.local_rank_count(), 4);
        assert_eq!(
            placement.owned_global_ranks().collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(placement.owns_global_root());
        assert!(placement.is_single_process());
    }

    #[test]
    fn node_local_slice_maps_local_to_global_ranks() {
        let placement = TensorParallelPlacement::new(8, 4, 4).unwrap();

        assert_eq!(
            placement.owned_global_ranks().collect::<Vec<_>>(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(placement.global_rank(0).unwrap(), 4);
        assert_eq!(placement.global_rank(3).unwrap(), 7);
        assert!(!placement.owns_global_root());
        assert!(!placement.is_single_process());
    }

    #[test]
    fn invalid_rank_ranges_are_rejected() {
        assert!(TensorParallelPlacement::single_process(0).is_err());
        assert!(TensorParallelPlacement::new(4, 0, 0).is_err());
        assert!(TensorParallelPlacement::new(4, 3, 2).is_err());
        assert!(TensorParallelPlacement::new(usize::MAX, usize::MAX, 1).is_err());
        assert!(
            TensorParallelPlacement::new(4, 2, 2)
                .unwrap()
                .global_rank(2)
                .is_err()
        );
    }
}
