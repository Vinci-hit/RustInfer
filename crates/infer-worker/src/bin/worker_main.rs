//! 旧版 ZMQ Worker server 入口已随 `SharedBuffers` / `WorkerServer` 一起下架。
//!
//! 当前 `worker` 子模块只暴露 `ModelRunner` 与 `BatchWorkspace` 这两件 GPU 侧
//! 基建；server / protocol 一层尚未基于新 Runner API 重写，因此这个 bin 暂时
//! 作为占位：构建可通过，但运行时直接拒绝启动，避免使用方误以为还能拉起 ZMQ
//! 服务。等 server 重写完成后再恢复完整 main。
fn main() {
    eprintln!(
        "rustinfer-worker: ZMQ server backend is currently disabled.\n\
         The old `WorkerServer` / `SharedBuffers` API has been removed and the\n\
         new server (built on top of `ModelRunner`) is not in tree yet.\n\
         For now, drive `ModelRunner` directly via the runner integration tests."
    );
    std::process::exit(2);
}
