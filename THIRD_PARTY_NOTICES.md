# Third-party notices

RustInfer includes source from third-party projects. Their licenses apply to
those components independently of RustInfer's Apache-2.0 license.

## FlashAttention

The CUDA backend vendors a selective, modified snapshot of the FlashAttention
Hopper implementation from release `v2.8.3` (commit
`060c9188beec3a8b62b33a3bfa6d5d2d44975fab`). It is distributed under the BSD
3-Clause License. See
[`crates/infer-backend-cuda/src/kernels/third_party/fa3/`](crates/infer-backend-cuda/src/kernels/third_party/fa3/)
for the license and detailed provenance.
