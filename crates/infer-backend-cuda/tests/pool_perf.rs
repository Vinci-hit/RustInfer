//! Manual eager CUDA pool benchmark using only the public memory interface.
//!
//! Run both revisions on the same GPU with the same build profile, for example:
//! `CUDA_VISIBLE_DEVICES=7 RUSTINFER_POOL_BENCH_LABEL=before cargo test -p infer-backend-cuda --test pool_perf -- --ignored --nocapture --test-threads=1`
//! Timings include the allocator's normal asynchronous zero-initialization and
//! the final compute-stream synchronization. Byte counts describe requests,
//! not the number or capacity of underlying CUDA allocations.

use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::device::MemoryPort;
use infer_core::exec::ExecScope;
use std::ptr::NonNull;
use std::time::Instant;

const MIB: usize = 1024 * 1024;
const TRACE_STEPS: usize = 16;
const WARMUP_REPEATS: usize = 2;
const ROUND_REPEATS: usize = 8;
const ROUNDS: usize = 5;
const BF16_BYTES: usize = 2;
const WIDTHS: [usize; 3] = [2048, 4096, 8192];

/// Preserve the requested size through Drop: a pool may internally round or
/// reuse a larger block, but MemoryPort::free_bytes takes the original size.
struct Allocation<'a> {
    device: &'a Cuda,
    ptr: NonNull<u8>,
    requested_bytes: usize,
}

impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        // SAFETY: this owns exactly one successful allocation on this device;
        // no pointer escapes the step, and its original byte count is retained.
        unsafe {
            self.device.free_bytes(self.ptr, self.requested_bytes);
        }
    }
}

fn step(device: &Cuda, tokens: usize) {
    // Keep all three projections live together before returning any to the
    // pool. Allocation's Drop also releases earlier blocks if a later one fails.
    let allocations = WIDTHS.map(|width| {
        let requested_bytes = tokens * width * BF16_BYTES;
        Allocation {
            device,
            ptr: device
                .alloc_bytes(requested_bytes)
                .expect("allocate eager projection scratch"),
            requested_bytes,
        }
    });
    drop(allocations);
}

fn run_trace(device: &Cuda, tokens: &[usize; TRACE_STEPS], repeats: usize) {
    for _ in 0..repeats {
        for &tokens in tokens {
            step(device, tokens);
        }
    }
}

fn run_case(label: &str, case: &str, tokens: &[usize; TRACE_STEPS]) {
    // Zero-sized fixed regions are supported by CudaMemoryPlan. Disable the
    // arena so every request exercises the eager pool, with no arena warmup.
    let device = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 0,
            graph_arena_bytes: 0,
            pool_retain_bytes: 256 * MIB,
        },
    )
    .expect("create CUDA device for pool benchmark");
    let scope = device.scope();
    let _active = scope.enter();

    run_trace(&device, tokens, WARMUP_REPEATS);
    let warm_stats = device.config.pool_stats();
    let steps_per_round = TRACE_STEPS * ROUND_REPEATS;
    let mut samples = [0.0_f64; ROUNDS];
    for sample in &mut samples {
        scope.synchronize().expect("synchronize before pool round");
        let started = Instant::now();
        run_trace(&device, tokens, ROUND_REPEATS);
        scope.synchronize().expect("synchronize after pool round");
        *sample = started.elapsed().as_secs_f64() * 1_000_000.0 / steps_per_round as f64;
    }
    let mut sorted = samples;
    sorted.sort_by(f64::total_cmp);
    let bytes_per_token = WIDTHS.iter().sum::<usize>() * BF16_BYTES;
    let requested_bytes_per_round = tokens.iter().sum::<usize>() * bytes_per_token * ROUND_REPEATS;
    let peak_live_bytes = tokens.iter().max().unwrap() * bytes_per_token;
    let csv_label = label.replace('"', "\"\"");
    println!(
        "\"{csv_label}\",{case},{:.3},{steps_per_round},{},{requested_bytes_per_round},{peak_live_bytes},{TRACE_STEPS},{WARMUP_REPEATS},{ROUND_REPEATS},{ROUNDS},{:.3},{:.3},{:.3},{:.3},{:.3}",
        sorted[ROUNDS / 2],
        steps_per_round * ROUNDS,
        samples[0],
        samples[1],
        samples[2],
        samples[3],
        samples[4],
    );
    let stats = device.config.pool_stats();
    println!(
        "pool_stats,{case},requests={},hits={},larger_reuses={},malloc_calls={},frees={},pooled_bytes={},peak_reserved_bytes={}",
        stats.allocation_requests - warm_stats.allocation_requests,
        stats.cache_hits - warm_stats.cache_hits,
        stats.larger_reuses - warm_stats.larger_reuses,
        stats.cuda_malloc_calls - warm_stats.cuda_malloc_calls,
        stats.cuda_frees - warm_stats.cuda_frees,
        stats.pooled_bytes,
        stats.peak_reserved_bytes,
    );
}

#[test]
#[ignore = "manual benchmark requiring a CUDA GPU"]
fn eager_pool_projection_traces() {
    let label = std::env::var("RUSTINFER_POOL_BENCH_LABEL").unwrap_or_else(|_| "unlabeled".into());
    println!(
        "label,case,median_us_per_step,steps_per_round,measured_steps,requested_bytes_per_round,peak_live_bytes,trace_steps,warmup_repeats,round_repeats,rounds,round_1_us_per_step,round_2_us_per_step,round_3_us_per_step,round_4_us_per_step,round_5_us_per_step"
    );
    run_case(&label, "fixed_4096", &[4096; TRACE_STEPS]);
    let descending = std::array::from_fn(|index| 4096 - index * 16);
    run_case(&label, "variable_4096_to_3856", &descending);
    let ascending = std::array::from_fn(|index| 3856 + index * 16);
    run_case(&label, "variable_3856_to_4096", &ascending);
    // Fixed permutation of the same 16 sizes makes both revisions replay an
    // identical mixed-order trace, without favoring best-fit in one direction.
    let permutation = [7, 0, 13, 4, 10, 2, 15, 6, 1, 12, 8, 3, 14, 5, 11, 9];
    let shuffled = permutation.map(|index| descending[index]);
    run_case(&label, "variable_fixed_shuffle", &shuffled);
}
