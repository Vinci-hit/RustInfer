//! CUDA graph scratch allocation and capture recovery integration tests.
//!
//! After loading the CUDA library environment, run on physical GPU 7:
//! `CUDA_VISIBLE_DEVICES=7 cargo test -p infer-backend-cuda --test graph_memory -- --ignored --test-threads=1`
//! Each test uses logical device 0 and a deliberately small graph arena.

use infer_backend_cuda::{Cuda, CudaMemoryPlan, CudaScope, ffi};
use infer_core::exec::ExecScope;
use infer_core::ports::DecodePipelineOps;
use infer_core::tensor::Tensor;

const MIB: usize = 1024 * 1024;
const ARENA_BYTES: usize = 512;
const LARGE_ELEMENTS: usize = 256;

fn scope() -> CudaScope {
    Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: MIB,
            graph_arena_bytes: ARENA_BYTES,
            pool_retain_bytes: MIB,
        },
    )
    .expect("create CUDA device with a small graph arena")
    .scope()
}

fn tensor(scope: &CudaScope, values: &[i32]) -> Tensor<i32, Cuda> {
    Tensor::from_host_slice(values, [values.len()], scope.device())
        .expect("upload persistent tensor")
}

/// Record through temporary arena storage whose Rust owner dies before capture ends.
fn record_copy(scope: &CudaScope, source: &Tensor<i32, Cuda>, output: &mut Tensor<i32, Cuda>) {
    let mut scratch =
        Tensor::zeros([source.numel()], scope.device()).expect("allocate graph scratch");
    enqueue_copy(scope, source, &mut scratch);
    enqueue_copy(scope, &scratch, output);
}

fn enqueue_copy(scope: &CudaScope, source: &Tensor<i32, Cuda>, output: &mut Tensor<i32, Cuda>) {
    assert!(source.is_contiguous() && output.is_contiguous());
    assert_eq!(source.numel(), output.numel());
    assert_ne!(source.data_ptr(), output.data_ptr());
    // Tensor::copy_from synchronizes today. Use a recordable D2D operation to
    // exercise graph ownership independently of that copy policy.
    // SAFETY: equally sized, contiguous test tensors own distinct storage on
    // this device. Persistent tensors and the arena survive capture/replay.
    let code = unsafe {
        ffi::cudaMemcpyAsync(
            output.data_ptr_mut().cast(),
            source.data_ptr().cast(),
            source.numel() * std::mem::size_of::<i32>(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            scope.device().config.stream,
        )
    };
    assert_eq!(code, ffi::cudaError_cudaSuccess, "record async D2D copy");
}

fn assert_replay(scope: &CudaScope, key: u64, output: &Tensor<i32, Cuda>, expected: &[i32]) {
    scope.graph_launch(key).expect("launch graph");
    scope.synchronize().expect("synchronize graph replay");
    assert_eq!(
        output.to_host_vec().expect("download graph output"),
        expected
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn capture_overflow_leaves_the_warm_pool_allocation_available_after_abort() {
    let scope = scope();
    let _active = scope.enter();
    let warm = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device()).expect("warm pool");
    let pool_address = warm.data_ptr();
    scope.synchronize().expect("finish pool warmup");
    drop(warm);

    scope.graph_capture_begin().expect("begin capture");
    let allocation = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device());
    scope
        .graph_capture_abort()
        .expect("abort overflowing capture");
    assert!(
        allocation.is_err(),
        "capture must reject a warm pool fallback"
    );

    let reused = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device())
        .expect("reuse pool after capture abort");
    assert_eq!(reused.data_ptr(), pool_address);
    assert_eq!(reused.to_host_vec().unwrap(), vec![0; LARGE_ELEMENTS]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn eager_pipeline_arena_overflow_still_reuses_the_pool() {
    let scope = scope();
    let _active = scope.enter();
    let warm = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device()).expect("warm pool");
    let pool_address = warm.data_ptr();
    scope.synchronize().expect("finish pool warmup");
    drop(warm);

    Cuda::pipeline_arena_begin(&scope).expect("begin eager arena session");
    let scratch = Tensor::<i32, Cuda>::zeros([4], scope.device()).expect("allocate eager scratch");
    let fallback = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device())
        .expect("eager overflow may fall back to the pool");
    Cuda::pipeline_arena_end(&scope);

    assert_eq!(fallback.data_ptr(), pool_address);
    assert_eq!(scratch.to_host_vec().unwrap(), vec![0; 4]);
    assert_eq!(fallback.to_host_vec().unwrap(), vec![0; LARGE_ELEMENTS]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn healthy_capture_abort_discards_recorded_work_and_allows_recapture() {
    let scope = scope();
    let _active = scope.enter();
    let mut source = tensor(&scope, &[1, 2, 3, 4]);
    let mut output = tensor(&scope, &[-1; 4]);
    scope
        .synchronize()
        .expect("finish persistent buffer uploads");

    scope
        .graph_capture_begin()
        .expect("begin capture to discard");
    record_copy(&scope, &source, &mut output);
    scope
        .graph_capture_abort()
        .expect("discard healthy capture");
    assert!(!scope.graph_ready(1));
    scope.synchronize().expect("synchronize after abort");
    assert_eq!(output.to_host_vec().unwrap(), vec![-1; 4]);

    scope
        .graph_capture_begin()
        .expect("begin replacement capture");
    record_copy(&scope, &source, &mut output);
    scope.graph_capture_end(1).expect("publish complete graph");
    assert!(scope.graph_ready(1));
    assert_eq!(output.to_host_vec().unwrap(), vec![-1; 4]);

    source.upload_from_host(&[17, 18, 19, 20]).unwrap();
    assert_replay(&scope, 1, &output, &[17, 18, 19, 20]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn swallowed_allocation_failure_cannot_publish_or_replace_a_graph() {
    let scope = scope();
    let _active = scope.enter();
    let original = tensor(&scope, &[1, 2, 3, 4]);
    let replacement = tensor(&scope, &[9, 8, 7, 6]);
    let mut output = tensor(&scope, &[-1; 4]);
    scope
        .synchronize()
        .expect("finish persistent buffer uploads");

    scope.graph_capture_begin().expect("capture original graph");
    record_copy(&scope, &original, &mut output);
    scope.graph_capture_end(1).expect("publish original graph");
    assert_replay(&scope, 1, &output, &[1, 2, 3, 4]);

    for key in [2, 1] {
        output.upload_from_host(&[-1; 4]).unwrap();
        scope.synchronize().expect("finish output reset");
        scope.graph_capture_begin().expect("begin failing capture");
        record_copy(&scope, &replacement, &mut output);
        let allocation = Tensor::<i32, Cuda>::zeros([LARGE_ELEMENTS], scope.device());
        // Deliberately call normal end after ignoring the allocation error.
        let ended = scope.graph_capture_end(key);
        assert!(allocation.is_err());
        assert!(ended.is_err(), "a locally failed capture must be discarded");
        assert!(!scope.graph_ready(2), "failed capture published a new key");
        assert!(
            scope.graph_ready(1),
            "failed replacement removed the old graph"
        );
        assert_eq!(output.to_host_vec().unwrap(), vec![-1; 4]);
        assert_replay(&scope, 1, &output, &[1, 2, 3, 4]);
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn nested_capture_and_arena_reset_preserve_the_original_recording() {
    let scope = scope();
    let _active = scope.enter();
    let first = tensor(&scope, &[1, 2, 3, 4]);
    let second = tensor(&scope, &[9, 8, 7, 6]);
    let mut first_output = tensor(&scope, &[-1; 4]);
    let mut second_output = tensor(&scope, &[-1; 4]);
    scope
        .synchronize()
        .expect("finish persistent buffer uploads");

    scope.graph_capture_begin().expect("begin original capture");
    let mut first_scratch = Tensor::zeros([4], scope.device()).expect("allocate first scratch");
    enqueue_copy(&scope, &first, &mut first_scratch);

    assert!(scope.graph_capture_begin().is_err());
    assert!(scope.device().config.capture_begin_relaxed().is_err());
    assert!(Cuda::pipeline_arena_begin(&scope).is_err());
    // A nested eager cleanup must also leave the original capture arena active.
    Cuda::pipeline_arena_end(&scope);

    let mut second_scratch = Tensor::zeros([4], scope.device()).expect("allocate second scratch");
    assert_ne!(first_scratch.data_ptr(), second_scratch.data_ptr());
    enqueue_copy(&scope, &second, &mut second_scratch);
    enqueue_copy(&scope, &first_scratch, &mut first_output);
    enqueue_copy(&scope, &second_scratch, &mut second_output);
    scope.graph_capture_end(1).expect("end original capture");
    drop(first_scratch);
    drop(second_scratch);

    assert_replay(&scope, 1, &first_output, &[1, 2, 3, 4]);
    assert_eq!(second_output.to_host_vec().unwrap(), vec![9, 8, 7, 6]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn abort_recovers_a_capture_invalidated_by_cuda() {
    invalidated_capture_recovers(true);
    invalidated_capture_recovers(false);
}

fn invalidated_capture_recovers(explicit_abort: bool) {
    let scope = scope();
    let _active = scope.enter();
    let source = tensor(&scope, &[13, 14, 15, 16]);
    let mut output = tensor(&scope, &[-1; 4]);
    scope
        .synchronize()
        .expect("finish persistent buffer uploads");

    scope
        .graph_capture_begin()
        .expect("begin capture to invalidate");
    record_copy(&scope, &source, &mut output);
    // Synchronizing a capturing stream is forbidden even in relaxed capture mode.
    let status = unsafe { ffi::cudaStreamSynchronize(scope.device().config.stream) };
    assert_ne!(status, ffi::cudaError_cudaSuccess);
    assert_eq!(scope.graph_debug_state(), "invalidated");
    if explicit_abort {
        scope
            .graph_capture_abort()
            .expect("abort CUDA-invalidated capture");
    } else {
        let error = scope
            .graph_capture_end(1)
            .expect_err("cannot publish invalidated capture");
        assert!(
            !error.is_fatal(),
            "capture invalidation is recoverable: {error}"
        );
    }
    assert!(!scope.graph_ready(1));
    scope
        .synchronize()
        .expect("stream recovers after invalidation");
    assert_eq!(output.to_host_vec().unwrap(), vec![-1; 4]);

    scope
        .graph_capture_begin()
        .expect("recapture after invalidation");
    record_copy(&scope, &source, &mut output);
    scope.graph_capture_end(1).expect("end recovered capture");
    assert_replay(&scope, 1, &output, &[13, 14, 15, 16]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn low_level_capture_without_an_arena_rejects_dynamic_allocation_and_recovers() {
    let scope = scope();
    let _active = scope.enter();
    let source = tensor(&scope, &[21, 22, 23, 24]);
    let mut output = tensor(&scope, &[-1; 4]);
    scope
        .synchronize()
        .expect("finish persistent buffer uploads");

    // Unlike ExecScope::graph_capture_begin, this does not open an arena session.
    scope
        .device()
        .config
        .capture_begin_relaxed()
        .expect("begin low-level capture");
    let allocation = Tensor::<i32, Cuda>::zeros([4], scope.device());
    scope
        .device()
        .config
        .capture_abort()
        .expect("abort capture without an arena");
    assert!(
        allocation.is_err(),
        "capture cannot use the eager allocator"
    );
    let eager =
        Tensor::<i32, Cuda>::zeros([4], scope.device()).expect("allocate eagerly after abort");
    assert_eq!(eager.to_host_vec().unwrap(), vec![0; 4]);

    // The existing low-level API remains valid for fully preallocated work.
    scope.device().config.capture_begin_relaxed().unwrap();
    enqueue_copy(&scope, &source, &mut output);
    scope.graph_capture_end(2).unwrap();
    assert_replay(&scope, 2, &output, &[21, 22, 23, 24]);

    scope
        .graph_capture_begin()
        .expect("begin capture with an arena");
    record_copy(&scope, &source, &mut output);
    scope
        .graph_capture_end(1)
        .expect("end capture with an arena");
    assert_replay(&scope, 1, &output, &[21, 22, 23, 24]);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn scoped_copy_is_capture_safe_and_rejects_partial_overlap() {
    use infer_core::ports::MathOps;
    let scope = scope();
    let mut source = tensor(&scope, &[1, 2, 3, 4]);
    let mut output = tensor(&scope, &[0, 0, 0, 0]);
    let same = source.clone();
    Cuda::copy_tensor(&scope, &same, &mut source).unwrap();
    assert!(
        Cuda::copy_tensor(
            &scope,
            &source.narrow(0, 0, 3).unwrap(),
            &mut source.narrow(0, 1, 3).unwrap(),
        )
        .is_err()
    );
    scope.synchronize().unwrap();
    scope.graph_capture_begin().unwrap();
    Cuda::copy_tensor(&scope, &source, &mut output).unwrap();
    scope.graph_capture_end(13).unwrap();
    assert_replay(&scope, 13, &output, &[1, 2, 3, 4]);
    source.upload_from_host(&[9, 8, 7, 6]).unwrap();
    assert_replay(&scope, 13, &output, &[9, 8, 7, 6]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn scope_timing_supports_replay_and_keeps_stream_alive() {
    let scope = scope();
    let input = tensor(&scope, &[7, 8, 9]);
    let mut output = tensor(&scope, &[0, 0, 0]);
    let mut timer = scope.create_timer().unwrap().expect("CUDA timer");
    scope.graph_capture_begin().unwrap();
    <Cuda as infer_core::ports::MathOps>::copy_tensor(&scope, &input, &mut output).unwrap();
    scope.graph_capture_end(7171).unwrap();
    timer.start().unwrap();
    scope.graph_launch(7171).unwrap();
    timer.stop().unwrap();
    // Polling may yield None while work is pending; only the test synchronizes.
    let _ = timer.elapsed_ms().unwrap();
    scope.synchronize().unwrap();
    let ms = timer.elapsed_ms().unwrap().expect("completed event");
    assert!(ms.is_finite() && ms >= 0.0);
    assert_eq!(output.to_host_vec().unwrap(), vec![7, 8, 9]);
    drop(input);
    drop(output);
    drop(scope);
    timer.start().unwrap();
    timer.stop().unwrap();
    // Timer owns its CUDA context through asynchronous destruction.
}
