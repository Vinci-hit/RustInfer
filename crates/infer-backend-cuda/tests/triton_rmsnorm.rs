//! GPU integration coverage with an independent CPU oracle.
//! Run with `--features triton --test triton_rmsnorm -- --ignored --test-threads=1`.
#![cfg(feature = "triton")]

include!("common/aot_rmsnorm.rs");

fn assert_aot_available(scope: &CudaScope) {
    assert!(
        scope.device().config.triton_available(),
        "these tests require triton kernels built for this GPU's CUDA_ARCH"
    );
}
