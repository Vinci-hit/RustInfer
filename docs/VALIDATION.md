# Request limits and validation

The chat and text completion endpoints acquire the configured HTTP admission
permit before reading the request body. Overload returns HTTP 429. The permit
remains held during inference, streaming, and non-cancellable background
preprocessing; extraction errors release it.

Chat bodies allow up to 56 MiB to accommodate base64 images. Chat text (the sum
of role strings and text parts) and completion text prompts are limited to
1 MiB of UTF-8 each, returning HTTP 400 when exceeded. Stop strings have a
separate aggregate 1 MiB limit. Image URLs do not count toward the text limit;
the existing image count, encoded size, pixel, and visual-token limits still
apply. Text completions retain Axum's default 2 MiB body limit. Prompt and stop
string tokenization run on blocking workers while retaining admission permits.

## CI

The normal CI workflow checks CPU crates and builds the Docker `builder` stage
on every push and pull request. This compiles and links the production CUDA
worker and kernels for `sm_90` without requiring a GPU. It does not publish an
image or establish numerical correctness.

Run the **GPU validation** workflow manually on a trusted revision using a
self-hosted Linux x64 runner labelled `gpu`, with Docker, NVIDIA Container
Toolkit, and an NVIDIA GPU. Choose the architecture matching that GPU. The
workflow builds the existing Docker builder environment and explicitly runs
the ignored BF16 Gated DeltaNet CPU/CUDA comparison across prefill and decode,
plus the HD256 paged-attention regression across split query tiles and cached
prefixes. The latter covers the 96 KiB shared-memory path used on Ada GPUs.
Do not use this persistent runner for untrusted pull requests.

To also run the ignored image preprocessing reference tests, enable
`vision_reference` and set repository variables `QWEN35_MODEL_PATH` and
`QWEN35_VISION_REFERENCE` to directories on the runner. The first contains the
model and tokenizer; the second contains prepared reference fixtures required
by `crates/infer-server/src/chat/multimodal.rs`. The reference directory must be
writable because the tests export the Rust patch bytes. These optional tests
compare preprocessing, not full model generation accuracy.
