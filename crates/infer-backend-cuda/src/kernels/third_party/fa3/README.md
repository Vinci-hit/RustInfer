# FlashAttention 3 provenance

This directory contains a selective, locally adapted snapshot of the
FlashAttention Hopper implementation:

- Upstream project: <https://github.com/Dao-AILab/flash-attention>
- Upstream release: `v2.8.3`
- Upstream commit: `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`
- Upstream source subtree: `hopper/`
- Imported into RustInfer: 2026-07-04

The upstream-derived material is redistributed under the BSD 3-Clause License
in [LICENSE](LICENSE). Existing upstream copyright notices are retained in the
source files. The snapshot is not byte-for-byte identical to upstream: it has
been reduced and adapted for RustInfer's torch-free, paged-KV BF16 forward path.
RustInfer-specific integration code and modifications remain covered by the
repository's Apache-2.0 license in addition to any upstream obligations.

`rustinfer_fa3_api.cu` is the local C ABI adapter. Review changes against the
commit above when refreshing this snapshot, retain the upstream license and
notices, and update this provenance record.
