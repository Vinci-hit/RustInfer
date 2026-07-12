# Changelog

All notable changes to RustInfer are documented here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-11

### Added

- Temperature, top-k, and top-p token sampling throughout the scheduling and
  worker path.
- Tokenized multi-token stop-sequence handling.
- Continuous browser SSE consumption with surfaced stream errors.
- Reproducible Rust toolchain metadata and CPU-safe continuous integration.
- FlashAttention 3 licensing and source provenance.

### Changed

- Disabled the unfinished diffusion API and runtime for this release. RustInfer
  1.0.1 supports LLM inference only.
- Made sample launch configurations and runtime-library discovery portable.
- Made the configured HTTP host effective and restricted cross-origin access by
  default.
- Unified workspace package metadata at version 1.0.1 and explicitly marked the
  in-repository crates as non-publishable.

### Fixed

- Validated tensor views and mutable storage ownership before exposing slices.
- Kept CUDA graph-captured FlashAttention scratch allocations alive and removed
  a pinned host-buffer reuse race.
- Validated completion token IDs before they reach embedding kernels.
- Sanitized rendered Markdown in the browser client.
- Bounded internal transport queues, returned ingestion failures promptly, and
  mapped request timeouts consistently.
- Removed generated Python bytecode, vendored frontend dependencies, a broken
  gitlink, private paths, and overly broad ignore rules from release sources.

## [1.0.0] - 2026-07-04

- Initial tagged RustInfer release.

[1.0.1]: https://github.com/Vinci-hit/RustInfer/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Vinci-hit/RustInfer/releases/tag/v1.0.0
