//! Run with CUDA_ARCH=sm_89 cargo test -p infer-backend-cuda --test bulk_upload -- --ignored --test-threads=1
use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::exec::ExecScope;
use infer_core::ports::MemoryPort;
use infer_core::tensor::Tensor;

#[test]
#[ignore = "requires a CUDA GPU"]
fn bulk_upload_preserves_bytes_across_slot_reuse_and_source_drop() {
    let device = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1024 * 1024,
            graph_arena_bytes: 1024,
            pool_retain_bytes: 1024 * 1024,
        },
    )
    .unwrap();
    let scope = device.scope();
    let _guard = scope.enter();
    // Includes empty/small inputs, exact chunk boundary, odd tail, and enough
    // chunks to reuse both slots multiple times with the default 8 MiB setting.
    for (round, size) in [
        0,
        17,
        8 * 1024 * 1024,
        40 * 1024 * 1024 + 17,
        24 * 1024 * 1024 + 3,
    ]
    .into_iter()
    .enumerate()
    {
        let mut source: Vec<u8> = (0..size)
            .map(|i| ((i * 13 + i / 4096 + round * 31) % 251) as u8)
            .collect();
        let tensor = Tensor::<i8, Cuda>::from_host_bytes(&source, [size], &device).unwrap();
        source.fill(0);
        drop(source);
        let actual = tensor.to_host_vec().unwrap();
        assert!(
            actual
                .iter()
                .enumerate()
                .all(|(i, &v)| v as u8 == ((i * 13 + i / 4096 + round * 31) % 251) as u8)
        );
    }
    // Reject synchronous host uploads before enqueueing into a captured graph.
    let dst = Tensor::<i8, Cuda>::zeros([17], &device).unwrap();
    device.synchronize().unwrap();
    scope.graph_capture_begin().unwrap();
    let result = unsafe {
        device.upload_bulk(
            std::ptr::NonNull::new(dst.data_ptr_mut().cast()).unwrap(),
            [1u8; 17].as_ptr(),
            17,
        )
    };
    assert!(result.unwrap_err().to_string().contains("graph capture"));
    scope.graph_capture_abort().unwrap();
    assert_eq!(dst.to_host_vec().unwrap(), vec![0; 17]);
}
