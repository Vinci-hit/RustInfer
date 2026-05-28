//! CudaGraphRunner — manages capture, storage, and replay of CUDA graphs.
//!
//! # Lifecycle
//!
//! ```text
//! 1. init(capture_sizes=[1,2,4,8,16])
//!     │
//!     ▼
//! 2. warmup: for each size, run dummy forward (eager) to init lazy params
//!     │
//!     ▼
//! 3. capture: for each size, capture the forward into a graph
//!     │
//!     ▼
//! 4. ready: graphs[1], graphs[2], graphs[4], graphs[8], graphs[16] cached
//!     │
//!     ▼
//! 5. serve:
//!     request(batch_size=5)
//!       → binary search → pick graphs[8] (next >= 5)
//!       → pad input to 8
//!       → replay graph
//!       → slice output[:5]
//!
//!     request(batch_size=20 > max=16)
//!       → fallback to eager mode
//! ```
//!
//! # Design (DDD perspective)
//!
//! CudaGraphRunner lives in the **app** layer (execution orchestration).
//! It borrows infra (CudaConfig for capture/replay) and domain (model forward).

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::Dtype;
use crate::domain::tensor::Tensor;
use crate::domain::model::{LlmModel, ForwardContext};

#[cfg(feature = "cuda")]
use crate::infrastructure::cuda::{Cuda, CudaConfig, config::GraphSlot};

/// State of a single graph slot (one captured batch size).
#[cfg(feature = "cuda")]
#[derive(Debug)]
enum SlotState {
    /// Not yet warmed up.
    Cold,
    /// Warmed up (eager ran), ready to capture.
    Warm,
    /// Captured and ready for replay.
    Captured { slot: GraphSlot },
}

/// CudaGraphRunner manages a set of pre-captured CUDA graphs
/// for different batch sizes, enabling zero-overhead kernel launch.
#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    /// Sorted list of capture sizes (e.g. [1, 2, 4, 8, 16]).
    capture_sizes: Vec<usize>,
    /// State for each capture size (indexed same as capture_sizes).
    states: Vec<SlotState>,
    /// Maximum batch size that has a graph. Requests above this use eager.
    max_capture_size: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    /// Create a new runner with the given capture sizes.
    /// Sizes must be sorted ascending and non-empty.
    pub fn new(capture_sizes: Vec<usize>) -> Self {
        assert!(!capture_sizes.is_empty(), "capture_sizes must not be empty");
        assert!(
            capture_sizes.windows(2).all(|w| w[0] < w[1]),
            "capture_sizes must be sorted ascending with no duplicates"
        );
        let max_capture_size = *capture_sizes.last().unwrap();
        let states = capture_sizes.iter().map(|_| SlotState::Cold).collect();
        Self { capture_sizes, states, max_capture_size }
    }

    /// Phase 1: Warmup — run each capture size in eager mode to initialize
    /// lazy parameters (cuBLAS algorithm selection, workspace, etc.).
    pub fn warmup<T: Dtype, D: OpBackend, M: LlmModel<T, D>>(
        &mut self,
        model: &M,
        make_dummy_inputs: &dyn Fn(usize) -> (Tensor<i32, D>, ForwardContext<'static, T, D>),
    ) -> OpResult<()> {
        for (i, &size) in self.capture_sizes.iter().enumerate() {
            // Run forward with dummy data (eager, no graph capture)
            let (input_ids, mut ctx) = make_dummy_inputs(size);
            let _ = model.forward(&input_ids, &mut ctx)?;
            self.states[i] = SlotState::Warm;
        }
        Ok(())
    }

    /// Phase 2: Capture — for each warmed-up size, capture into a CUDA graph.
    pub fn capture<T: Dtype, M: LlmModel<T, Cuda>>(
        &mut self,
        model: &M,
        config: &CudaConfig,
        make_dummy_inputs: &dyn Fn(usize) -> (Tensor<i32, Cuda>, ForwardContext<'static, T, Cuda>),
    ) -> OpResult<()> {
        for (i, &size) in self.capture_sizes.iter().enumerate() {
            match self.states[i] {
                SlotState::Warm => {}
                _ => continue, // skip cold or already captured
            }

            let slot = GraphSlot::LlmDecode {
                batch: size,
                buffer_id: 0,
                slot_signature: size as u64,
            };

            // Begin capture (relaxed mode for cuBLAS compatibility)
            config.capture_begin_relaxed()?;

            // Run the forward — all GPU ops are recorded, not executed
            let (input_ids, mut ctx) = make_dummy_inputs(size);
            let _ = model.forward(&input_ids, &mut ctx)?;

            // End capture — instantiate the graph
            config.capture_end(slot)?;
            self.states[i] = SlotState::Captured { slot };
        }
        Ok(())
    }

    /// Serve a request: find the best graph and replay, or fall back to eager.
    ///
    /// Returns `GraphDecision` indicating how to execute.
    pub fn decide(&self, batch_size: usize) -> GraphDecision {
        if batch_size > self.max_capture_size {
            return GraphDecision::Eager;
        }
        // Binary search for the smallest capture_size >= batch_size
        let idx = match self.capture_sizes.binary_search(&batch_size) {
            Ok(exact) => exact,
            Err(insert_point) => {
                if insert_point >= self.capture_sizes.len() {
                    return GraphDecision::Eager;
                }
                insert_point
            }
        };
        let padded_size = self.capture_sizes[idx];
        match &self.states[idx] {
            SlotState::Captured { slot } => GraphDecision::Replay {
                slot: *slot,
                padded_size,
                original_size: batch_size,
            },
            _ => GraphDecision::Eager, // not yet captured
        }
    }

    /// Execute a replay decision against the CudaConfig.
    pub fn replay(&self, config: &CudaConfig, decision: &GraphDecision) -> OpResult<()> {
        match decision {
            GraphDecision::Replay { slot, .. } => config.launch(*slot),
            GraphDecision::Eager => Ok(()), // caller handles eager path
        }
    }

    /// Invalidate all captured graphs (e.g. when KV cache grows).
    pub fn invalidate_all(&mut self, config: &CudaConfig) {
        config.invalidate_all_graphs();
        for state in &mut self.states {
            if matches!(state, SlotState::Captured { .. }) {
                *state = SlotState::Warm; // can re-capture
            }
        }
    }

    /// Check if all sizes are captured and ready.
    pub fn is_fully_ready(&self) -> bool {
        self.states.iter().all(|s| matches!(s, SlotState::Captured { .. }))
    }

    /// Get the list of capture sizes.
    pub fn capture_sizes(&self) -> &[usize] {
        &self.capture_sizes
    }

    /// Maximum batch size that has a captured graph.
    pub fn max_capture_size(&self) -> usize {
        self.max_capture_size
    }

    /// Lookup the captured slot for a specific size, if any. Used by the
    /// runner to know whether `replay()` can be issued.
    pub fn captured_slot_for(&self, size: usize) -> Option<GraphSlot> {
        self.capture_sizes.iter().position(|&s| s == size).and_then(|i| {
            match self.states[i] {
                SlotState::Captured { slot } => Some(slot),
                _ => None,
            }
        })
    }

    /// Reverse-order warmup + capture sweep.
    ///
    /// For each `size` in `capture_sizes`, in reverse (largest first):
    ///   1. Run `run_fn(size, false)` `warmup_passes` times (eager) —
    ///      initializes lazy cuBLAS / cuDNN algos AND uploads plan data.
    ///   2. `capture_begin_relaxed` → `run_fn(size, true)` → `capture_end`.
    ///      When `is_capture=true`, the closure must ONLY run forward +
    ///      argmax (no H2D memcpy), so the graph contains only kernels.
    ///   3. `launch` once to validate the graph.
    ///
    /// Splitting warmup vs capture ensures H2D memcpy operations stay
    /// OUTSIDE the captured graph. The device buffers already hold valid
    /// data from warmup, so the capture pass can run forward against
    /// those stable addresses without re-uploading.
    pub fn warmup_and_capture_all<F>(
        &mut self,
        config: &CudaConfig,
        warmup_passes: usize,
        mut run_fn: F,
    ) -> OpResult<()>
    where
        F: FnMut(usize, bool) -> OpResult<()>,
    {
        // Iterate sizes in REVERSE order (largest first) for memory-friendly
        // capture: cuBLASLt/cuDNN allocate more workspace for bigger sizes,
        // and we'd rather see those allocations before smaller-size graphs.
        let order: Vec<(usize, usize)> = self.capture_sizes.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .rev()
            .collect();

        for (i, size) in order {
            // 1. Warmup passes (eager) — includes H2D upload.
            for _ in 0..warmup_passes {
                run_fn(size, false)?;
            }
            // Wait for warmup kernels to finish before capturing — capture
            // shouldn't share a stream with in-flight kernels.
            config.synchronize()?;
            self.states[i] = SlotState::Warm;

            // 2. Capture — forward + argmax only, NO H2D memcpy.
            let slot = GraphSlot::LlmDecode {
                batch: size,
                buffer_id: 0,
                slot_signature: size as u64,
            };
            config.capture_begin_relaxed()?;
            // Run forward + argmax inside the captured region. If it errors,
            // we still need to call capture_end to leave the stream in a
            // sane state.
            let res = run_fn(size, true);
            if let Err(e) = res {
                // Best-effort: end capture into a throwaway graph and drop.
                let mut graph: crate::infrastructure::cuda::ffi::cudaGraph_t = std::ptr::null_mut();
                unsafe {
                    let _ = crate::infrastructure::cuda::ffi::cudaStreamEndCapture(config.stream, &mut graph);
                    if !graph.is_null() {
                        crate::infrastructure::cuda::ffi::cudaGraphDestroy(graph);
                    }
                }
                return Err(e);
            }
            config.capture_end(slot)?;
            self.states[i] = SlotState::Captured { slot };

            // 3. Launch once (validates the graph; also "primes" replay).
            config.launch(slot)?;
            config.synchronize()?;
        }
        Ok(())
    }
}

/// Decision from `CudaGraphRunner::decide()`.
#[derive(Debug, Clone)]
pub enum GraphDecision {
    /// Replay a captured graph. Pad input to `padded_size`, slice output to `original_size`.
    Replay {
        slot: GraphSlot,
        padded_size: usize,
        original_size: usize,
    },
    /// No suitable graph — run in eager mode (no graph, direct kernel launches).
    Eager,
}

impl GraphDecision {
    /// Whether this decision requires padding.
    pub fn needs_padding(&self) -> bool {
        matches!(self, GraphDecision::Replay { padded_size, original_size, .. } if padded_size != original_size)
    }

    /// The batch size to actually execute with (padded or original).
    pub fn execution_size(&self) -> usize {
        match self {
            GraphDecision::Replay { padded_size, .. } => *padded_size,
            GraphDecision::Eager => 0, // caller uses actual batch_size
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn decision_binary_search() {
        let mut runner = CudaGraphRunner::new(vec![1, 2, 4, 8, 16]);
        // Simulate all captured
        for (i, &size) in runner.capture_sizes.clone().iter().enumerate() {
            runner.states[i] = SlotState::Captured {
                slot: GraphSlot::LlmDecode { batch: size, buffer_id: 0, slot_signature: size as u64 },
            };
        }

        // Exact match
        match runner.decide(4) {
            GraphDecision::Replay { padded_size, original_size, .. } => {
                assert_eq!(padded_size, 4);
                assert_eq!(original_size, 4);
            }
            _ => panic!("expected Replay"),
        }

        // Needs padding: 5 → next is 8
        match runner.decide(5) {
            GraphDecision::Replay { padded_size, original_size, .. } => {
                assert_eq!(padded_size, 8);
                assert_eq!(original_size, 5);
            }
            _ => panic!("expected Replay"),
        }

        // Over max → eager
        assert!(matches!(runner.decide(20), GraphDecision::Eager));

        // Batch size 1 → exact
        match runner.decide(1) {
            GraphDecision::Replay { padded_size, .. } => assert_eq!(padded_size, 1),
            _ => panic!("expected Replay"),
        }
    }

    #[test]
    fn needs_padding() {
        let d = GraphDecision::Replay {
            slot: GraphSlot::LlmDecode { batch: 8, buffer_id: 0, slot_signature: 8 },
            padded_size: 8,
            original_size: 5,
        };
        assert!(d.needs_padding());

        let d2 = GraphDecision::Replay {
            slot: GraphSlot::LlmDecode { batch: 4, buffer_id: 0, slot_signature: 4 },
            padded_size: 4,
            original_size: 4,
        };
        assert!(!d2.needs_padding());
    }
}
