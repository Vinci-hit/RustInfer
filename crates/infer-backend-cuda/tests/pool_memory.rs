//! CUDA eager-pool ownership, capacity accounting, and capture isolation.
//!
//! `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-backend-cuda --test pool_memory -- --ignored --test-threads=1`
//! Tests use logical device 0 and small allocations; none forces device OOM.

use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope, ffi};
use infer_core::exec::ExecScope;
use infer_core::tensor::Tensor;

const MIB: usize = 1024 * 1024;
const LARGE_BYTES: usize = 8192;
const REUSE_BYTES: usize = 7680;
const OUTSIDE_SLACK_BYTES: usize = 7168;
const I32_BYTES: usize = std::mem::size_of::<i32>();

fn scope(pool_retain_bytes: usize) -> CudaScope {
    Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 0,
            graph_arena_bytes: 512,
            pool_retain_bytes,
        },
    )
    .expect("create CUDA device with a small memory plan")
    .scope()
}

fn zeros(scope: &CudaScope, bytes: usize) -> Tensor<i32, Cuda> {
    assert_eq!(bytes % I32_BYTES, 0);
    Tensor::zeros([bytes / I32_BYTES], scope.device()).expect("allocate eager tensor")
}

fn filled(scope: &CudaScope, bytes: usize, value: i32) -> Tensor<i32, Cuda> {
    assert_eq!(bytes % I32_BYTES, 0);
    let values = vec![value; bytes / I32_BYTES];
    Tensor::from_host_slice(&values, [values.len()], scope.device()).expect("upload tensor")
}

fn assert_values(tensor: &Tensor<i32, Cuda>, expected: i32) {
    assert_eq!(
        tensor.to_host_vec().expect("download tensor"),
        vec![expected; tensor.numel()],
    );
}

fn enqueue_copy(scope: &CudaScope, source: &Tensor<i32, Cuda>, output: &mut Tensor<i32, Cuda>) {
    assert_eq!(source.numel(), output.numel());
    // SAFETY: these contiguous tensors own separate, equally sized buffers on
    // the scope's device. Capture records the copy without executing it.
    let code = unsafe {
        ffi::cudaMemcpyAsync(
            output.data_ptr_mut().cast(),
            source.data_ptr().cast(),
            source.numel() * I32_BYTES,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            scope.device().config.stream,
        )
    };
    assert_eq!(code, ffi::cudaError_cudaSuccess, "record D2D copy");
}

macro_rules! assert_fields_unchanged {
    ($before:ident, $after:ident, $($field:ident),+ $(,)?) => {
        $(assert_eq!($before.$field, $after.$field, stringify!($field));)+
    };
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn larger_block_reuse_zeroes_each_request_and_preserves_capacity() {
    let scope = scope(MIB);
    let _active = scope.enter();
    let initial = scope.device().config.pool_stats();
    let large = filled(&scope, LARGE_BYTES, 71);
    let address = large.data_ptr();
    scope.synchronize().expect("finish nonzero upload");
    drop(large);
    let warm = scope.device().config.pool_stats();
    assert_eq!(warm.live_bytes, 0);
    assert_eq!(warm.pooled_bytes, LARGE_BYTES);
    assert_eq!(warm.reserved_bytes, LARGE_BYTES);

    let mut smaller = zeros(&scope, REUSE_BYTES);
    assert_eq!(smaller.data_ptr(), address);
    assert_values(&smaller, 0);
    let reused = scope.device().config.pool_stats();
    assert_eq!(reused.live_bytes, LARGE_BYTES);
    assert_eq!(reused.pooled_bytes, 0);
    assert_eq!(reused.reserved_bytes, LARGE_BYTES);
    assert_eq!(reused.cache_hits, warm.cache_hits + 1);
    assert_eq!(reused.larger_reuses, warm.larger_reuses + 1);
    assert_eq!(reused.cuda_malloc_calls, warm.cuda_malloc_calls);
    smaller
        .upload_from_host(&vec![29; REUSE_BYTES / I32_BYTES])
        .expect("overwrite reused prefix");
    scope.synchronize().expect("finish reused tensor upload");
    drop(smaller);
    let returned = scope.device().config.pool_stats();
    assert_eq!(returned.live_bytes, 0);
    assert_eq!(returned.pooled_bytes, LARGE_BYTES);
    assert_eq!(returned.reserved_bytes, LARGE_BYTES);

    let large_again = zeros(&scope, LARGE_BYTES);
    assert_eq!(large_again.data_ptr(), address);
    assert_values(&large_again, 0);
    drop(large_again);
    let final_stats = scope.device().config.pool_stats();
    assert_eq!(
        final_stats.allocation_requests,
        initial.allocation_requests + 3
    );
    assert_eq!(final_stats.cache_hits, initial.cache_hits + 2);
    assert_eq!(final_stats.larger_reuses, initial.larger_reuses + 1);
    assert_eq!(final_stats.cuda_malloc_calls, initial.cuda_malloc_calls + 1);
    assert_eq!(final_stats.cold_allocations, initial.cold_allocations + 1);
    assert_eq!(final_stats.live_bytes, 0);
    assert_eq!(final_stats.pooled_bytes, LARGE_BYTES);
    assert_eq!(final_stats.reserved_bytes, LARGE_BYTES);
    assert_eq!(final_stats.peak_live_bytes, LARGE_BYTES);
    assert_eq!(final_stats.peak_reserved_bytes, LARGE_BYTES);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn outside_slack_allocates_exact_capacity_and_live_blocks_do_not_overlap() {
    let scope = scope(MIB);
    let _active = scope.enter();
    let large = filled(&scope, LARGE_BYTES, 41);
    let warm_address = large.data_ptr();
    scope.synchronize().expect("finish pool warmup");
    drop(large);
    let before = scope.device().config.pool_stats();

    let smaller = zeros(&scope, OUTSIDE_SLACK_BYTES);
    assert_ne!(smaller.data_ptr(), warm_address);
    let after_cold = scope.device().config.pool_stats();
    assert_eq!(after_cold.cuda_malloc_calls, before.cuda_malloc_calls + 1);
    assert_eq!(after_cold.cold_allocations, before.cold_allocations + 1);
    assert_eq!(after_cold.cache_hits, before.cache_hits);
    assert_eq!(after_cold.live_bytes, OUTSIDE_SLACK_BYTES);
    assert_eq!(after_cold.pooled_bytes, LARGE_BYTES);
    assert_eq!(after_cold.reserved_bytes, LARGE_BYTES + OUTSIDE_SLACK_BYTES);

    let large_again = zeros(&scope, LARGE_BYTES);
    assert_eq!(large_again.data_ptr(), warm_address);
    let small_start = smaller.data_ptr() as usize;
    let large_start = large_again.data_ptr() as usize;
    assert!(
        small_start + OUTSIDE_SLACK_BYTES <= large_start
            || large_start + LARGE_BYTES <= small_start,
        "simultaneously live allocations must not overlap",
    );
    assert_values(&smaller, 0);
    assert_values(&large_again, 0);
    let both_live = scope.device().config.pool_stats();
    assert_eq!(both_live.live_bytes, LARGE_BYTES + OUTSIDE_SLACK_BYTES);
    assert_eq!(both_live.pooled_bytes, 0);
    assert_eq!(both_live.cuda_malloc_calls, after_cold.cuda_malloc_calls);
    drop(smaller);
    drop(large_again);
    let returned = scope.device().config.pool_stats();
    assert_eq!(returned.live_bytes, 0);
    assert_eq!(returned.pooled_bytes, LARGE_BYTES + OUTSIDE_SLACK_BYTES);
    assert_eq!(returned.reserved_bytes, returned.pooled_bytes);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn trim_frees_only_idle_capacity_and_preserves_live_data() {
    let scope = scope(MIB);
    let _active = scope.enter();
    let live = filled(&scope, 4096, 17);
    let idle = filled(&scope, LARGE_BYTES, 23);
    scope.synchronize().expect("finish live and idle uploads");
    drop(idle);
    let before = scope.device().config.pool_stats();
    assert_eq!(before.live_bytes, 4096);
    assert_eq!(before.pooled_bytes, LARGE_BYTES);

    let freed = scope
        .device()
        .config
        .trim_pool(4096)
        .expect("trim idle pool");
    assert_eq!(freed, LARGE_BYTES);
    let trimmed = scope.device().config.pool_stats();
    assert_eq!(trimmed.live_bytes, 4096);
    assert_eq!(trimmed.pooled_bytes, 0);
    assert_eq!(trimmed.reserved_bytes, 4096);
    assert_eq!(trimmed.cuda_frees, before.cuda_frees + 1);
    assert_values(&live, 17);

    drop(live);
    assert_eq!(scope.device().config.trim_pool(0).unwrap(), 4096);
    let empty = scope.device().config.pool_stats();
    assert_eq!(empty.live_bytes, 0);
    assert_eq!(empty.pooled_bytes, 0);
    assert_eq!(empty.reserved_bytes, 0);
    assert_eq!(scope.device().config.trim_pool(0).unwrap(), 0);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn retention_limit_uses_reused_block_capacity_not_request_size() {
    const RETAIN_BYTES: usize = 12288;
    const OTHER_BYTES: usize = 4608;
    let scope = scope(RETAIN_BYTES);
    let _active = scope.enter();
    let large = zeros(&scope, LARGE_BYTES);
    let address = large.data_ptr();
    scope.synchronize().expect("finish retention warmup");
    drop(large);
    let reused = zeros(&scope, REUSE_BYTES);
    assert_eq!(reused.data_ptr(), address);
    let other = zeros(&scope, OTHER_BYTES);
    scope
        .synchronize()
        .expect("finish simultaneous allocations");
    drop(other);
    let before = scope.device().config.pool_stats();
    assert_eq!(before.live_bytes, LARGE_BYTES);
    assert_eq!(before.pooled_bytes, OTHER_BYTES);
    drop(reused);

    // Requested bytes would total exactly 12288; actual capacities total 12800.
    // Either block may be evicted, but retaining both would exceed the limit.
    let after = scope.device().config.pool_stats();
    assert_eq!(after.live_bytes, 0);
    assert!(after.pooled_bytes <= RETAIN_BYTES);
    assert!(after.pooled_bytes == OTHER_BYTES || after.pooled_bytes == LARGE_BYTES);
    assert_eq!(after.reserved_bytes, after.pooled_bytes);
    assert_eq!(after.cuda_frees, before.cuda_frees + 1);
    assert_eq!(
        scope.device().config.trim_pool(0).unwrap(),
        after.pooled_bytes
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn trim_during_capture_is_rejected_without_changing_the_pool() {
    let scope = scope(MIB);
    let _active = scope.enter();
    let warm = zeros(&scope, LARGE_BYTES);
    scope.synchronize().expect("finish pool warmup");
    drop(warm);
    let before = scope.device().config.pool_stats();

    scope.graph_capture_begin().expect("begin capture");
    let trim_result = scope.device().config.trim_pool(0);
    let during = scope.device().config.pool_stats();
    scope
        .graph_capture_abort()
        .expect("abort capture after trim rejection");
    assert!(trim_result.is_err());
    assert_fields_unchanged!(
        before,
        during,
        live_bytes,
        pooled_bytes,
        reserved_bytes,
        pending_bytes,
        cuda_malloc_calls,
        cuda_frees,
        allocation_requests,
    );
    assert_eq!(scope.device().config.trim_pool(0).unwrap(), LARGE_BYTES);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn capture_defers_eager_free_discards_the_graph_and_then_recovers() {
    let scope = scope(0);
    let _active = scope.enter();
    let source = filled(&scope, LARGE_BYTES, 19);
    let mut output = filled(&scope, LARGE_BYTES, -1);
    scope
        .synchronize()
        .expect("finish persistent tensor uploads");
    let before = scope.device().config.pool_stats();

    scope
        .graph_capture_begin()
        .expect("begin capture using eager storage");
    enqueue_copy(&scope, &source, &mut output);
    drop(source);
    let deferred = scope.device().config.pool_stats();
    let ended = scope.graph_capture_end(1);
    assert!(
        ended.is_err(),
        "a graph referencing released eager storage must be discarded"
    );
    assert!(!scope.graph_ready(1));
    assert_eq!(deferred.pending_bytes, before.pending_bytes + LARGE_BYTES);
    assert_eq!(deferred.live_bytes, before.live_bytes - LARGE_BYTES);
    assert_eq!(deferred.reserved_bytes, before.reserved_bytes);
    assert_eq!(deferred.cuda_frees, before.cuda_frees);
    assert_eq!(deferred.pooled_bytes, 0);

    let cleaned = scope.device().config.pool_stats();
    assert_eq!(cleaned.pending_bytes, 0);
    assert_eq!(cleaned.live_bytes, LARGE_BYTES);
    assert_eq!(cleaned.pooled_bytes, 0);
    assert_eq!(cleaned.reserved_bytes, LARGE_BYTES);
    assert_eq!(cleaned.cuda_frees, before.cuda_frees + 1);
    scope
        .synchronize()
        .expect("stream usable after deferred free cleanup");
    assert_values(&output, -1);

    let replacement = filled(&scope, LARGE_BYTES, 37);
    scope.synchronize().expect("finish replacement upload");
    scope
        .graph_capture_begin()
        .expect("begin recovered capture");
    enqueue_copy(&scope, &replacement, &mut output);
    scope.graph_capture_end(2).expect("publish recovered graph");
    scope.graph_launch(2).expect("launch recovered graph");
    scope.synchronize().expect("finish recovered graph");
    assert_values(&output, 37);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn graph_arena_allocations_do_not_change_eager_pool_statistics() {
    let scope = scope(MIB);
    let _active = scope.enter();
    let before = scope.device().config.pool_stats();
    scope
        .graph_capture_begin()
        .expect("begin arena-only capture");
    let arena_tensor = zeros(&scope, 256);
    drop(arena_tensor);
    scope
        .graph_capture_end(1)
        .expect("publish arena-only graph");
    scope.graph_launch(1).expect("replay arena-only graph");
    scope.synchronize().expect("finish arena graph");
    let after = scope.device().config.pool_stats();
    assert_fields_unchanged!(
        before,
        after,
        allocation_requests,
        cache_hits,
        larger_reuses,
        cuda_malloc_calls,
        cold_allocations,
        cuda_frees,
        allocation_retries,
        live_bytes,
        pooled_bytes,
        reserved_bytes,
        peak_live_bytes,
        peak_reserved_bytes,
        pending_bytes,
    );
}
