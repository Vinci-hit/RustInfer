//! Thread-local CUDA stream management.
//!
//! 仿 PyTorch 的 `torch.cuda.current_stream()` 设计：
//! - 每个线程维护一个 "当前 stream"
//! - model forward 入口通过 `with_cuda_stream` 设置
//! - 所有 Tensor 运算符和 Op 自动从 thread-local 取 stream
//! - 未设置时 fallback 到 default stream (null)

use std::cell::Cell;

use super::ffi::cudaStream_t;

thread_local! {
    static CURRENT_CUDA_STREAM: Cell<cudaStream_t> = const { Cell::new(std::ptr::null_mut()) };
}

/// 获取当前线程的 CUDA stream。
/// 未设置时返回 null（即 CUDA default stream）。
#[inline]
pub fn get_current_cuda_stream() -> cudaStream_t {
    CURRENT_CUDA_STREAM.with(|s| s.get())
}

/// 设置当前线程的 CUDA stream 并执行闭包，闭包结束后恢复旧值。
///
/// ```ignore
/// let cfg = CudaConfig::new()?;
/// with_cuda_stream(cfg.stream, || {
///     // 此 scope 内所有 Tensor 运算符自动使用 cfg.stream
///     let result = &latents + &(&noise_pred * dt);
/// });
/// ```
#[inline]
pub fn with_cuda_stream<F, R>(stream: cudaStream_t, f: F) -> R
where
    F: FnOnce() -> R,
{
    CURRENT_CUDA_STREAM.with(|s| {
        let old = s.replace(stream);
        let result = f();
        s.set(old);
        result
    })
}
