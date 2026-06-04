//! `gather_merge_input` — single dependency-merge of the bubble-free decode
//! pipeline.
//!
//! Each decode step needs its `input_ids` (buffer **A**) assembled from two
//! sources, only one of which is GPU-dependent:
//!
//! - **C** (`argmax_out_dev`): the previous step's on-device argmax output —
//!   the next token for every *continuing* sequence. Never leaves the device.
//! - **B** (`new_token_dev`): first tokens of *newly-admitted* sequences,
//!   uploaded by the CPU on a copy stream while the GPU forwards.
//!
//! `src[i]` selects row `i`'s source:
//!
//! ```text
//!   A[i] = src[i] >= 0 ? C[src[i]] : B[-src[i] - 1]
//! ```
//!
//! This kernel is the *only* synchronization point between the previous
//! step's output and the next step's input, replacing the per-step D2H
//! token read. With A read-only during forward, A can be downloaded for the
//! scheduler concurrently with the next forward — no bubble.

use crate::domain::ports::OpResult;
use crate::domain::tensor::Tensor;
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn gather_merge_input(
        a: *mut i32,
        c: *const i32,
        b: *const i32,
        src: *const i32,
        batch: i32,
        stream: cudaStream_t,
    );
}

/// Assemble next-step input ids `A` from previous output `C` and new-admit
/// tokens `B` per the selector `src`. Launches on `stream` (caller chooses
/// the compute stream). Zero allocation, zero D2H — graph-capturable.
///
/// Buffers are device i32:
/// - `a_out`  len ≥ `batch` (written)
/// - `c_prev` len ≥ max continuing `src[i]` + 1
/// - `b_new`  len ≥ number of negative `src` entries
/// - `src`    len ≥ `batch`
pub fn gather_merge_input_into(
    a_out: &mut Tensor<i32, Cuda>,
    c_prev: &Tensor<i32, Cuda>,
    b_new: &Tensor<i32, Cuda>,
    src: &Tensor<i32, Cuda>,
    batch: usize,
    stream: cudaStream_t,
) -> OpResult<()> {
    if batch == 0 {
        return Ok(());
    }
    // SAFETY: all four pointers are device buffers of the documented minimum
    // length; the kernel does one bounds-checked thread per row.
    unsafe {
        gather_merge_input(
            a_out.data_ptr_mut(),
            c_prev.data_ptr(),
            b_new.data_ptr(),
            src.data_ptr(),
            batch as i32,
            stream,
        );
    }
    Ok(())
}
