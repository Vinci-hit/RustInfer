//! Direct file reads into pinned host storage, followed by synchronous H2D.
use infer_backend_cuda::{Cuda, CudaMemoryPlan};
use infer_core::{exec::ExecScope, ports::MemoryPort, tensor::Tensor};
use std::os::unix::fs::FileExt;

#[test]
#[ignore = "requires a CUDA GPU"]
fn file_reads_fill_upload_source_directly_and_source_is_reusable_after_upload() {
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
    let len = 1024 * 1024 + 17;
    let expected: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!("rustinfer-pinned-read-{}", std::process::id()));
    std::fs::write(&path, &expected).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let buffer = device.alloc_host_buffer(len).unwrap();
    let mut buffer = std::thread::spawn(move || {
        let mut buffer = buffer;
        file.read_exact_at(buffer.bytes_mut(), 0).unwrap();
        buffer
    })
    .join()
    .unwrap();
    let tensor = Tensor::<i8, Cuda>::from_host_bytes(buffer.bytes(), [len], &device).unwrap();
    buffer.bytes_mut().fill(0);
    drop(buffer);
    let actual = tensor.to_host_vec().unwrap();
    assert!(actual.iter().zip(&expected).all(|(&a, &b)| a as u8 == b));
}
