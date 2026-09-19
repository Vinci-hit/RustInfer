//! GPU integration coverage with an independent CPU oracle.
//! Run with `--features cute-dsl --test cute_dsl_rmsnorm -- --ignored --test-threads=1`.
#![cfg(feature = "cute-dsl")]

include!("common/aot_rmsnorm.rs");

fn assert_aot_available(scope: &CudaScope) {
    assert!(
        scope.device().config.cute_dsl_available(),
        "these tests require cute_dsl kernels built for this GPU's CUDA_ARCH"
    );
}
