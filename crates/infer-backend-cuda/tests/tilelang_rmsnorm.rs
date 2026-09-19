//! GPU integration coverage with an independent CPU oracle.
//! Run with `--features tilelang --test tilelang_rmsnorm -- --ignored --test-threads=1`.
#![cfg(feature = "tilelang")]

include!("common/aot_rmsnorm.rs");

fn assert_aot_available(scope: &CudaScope) {
    assert!(
        scope.device().config.tilelang_available(),
        "these tests require tilelang kernels built for this GPU's CUDA_ARCH"
    );
}
