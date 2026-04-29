"""vllm-omni offline benchmark for Z-Image-Turbo at 256x256, 9 inference steps.

Mirrors RustInfer's `test_pipeline_bench_cuda_9step` as closely as possible
(same model dir, same resolution, same step count, same prompt) so the two
numbers can be compared apples-to-apples.

Run with the vllm-omni venv:
    /root/vllm-omni/.venv/bin/python scripts/bench_vllm_omni_z_image.py

Why vllm-omni's venv: the `/root/vllm_test/.venv` ships stock vLLM 0.19 which
does **not** carry the `vllm_omni` package; only the vllm-omni repo-local
venv at `/root/vllm-omni/.venv` has the editable install.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


# Match the Rust bench verbatim — same prompt, same seed as
# test_pipeline_bench_cuda_9step in crates/infer-worker/.../pipeline.rs.
PROMPT = (
    "a photograph of a cat wearing a red hat, sitting on a "
    "wooden bench in a sunny park"
)
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/z-image-turbo",
                   help="Local path to Z-Image-Turbo weights.")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-inference-steps", type=int, default=9,
                   help="Denoising steps. 2 = Turbo default, 9 = official full.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup iterations before measured run.")
    p.add_argument("--iters", type=int, default=5,
                   help="Number of measured iterations.")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    return p.parse_args()


def build_omni(model: str) -> Omni:
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=1,
        ring_degree=1,
        ulysses_mode="strict",
        cfg_parallel_size=1,
        tensor_parallel_size=1,
        vae_patch_parallel_size=1,
        enable_expert_parallel=False,
    )
    return Omni(
        model=model,
        mode="text-to-image",
        parallel_config=parallel_config,
        enforce_eager=False,     # keep CUDA graph on, matches our Rust path
        enable_cpu_offload=False,
        log_stats=False,
        enable_diffusion_pipeline_profiler=False,
    )


def make_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    # Fresh generator each call so the seed is reproducible.
    gen = torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED)
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        generator=gen,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=1,
        extra_args={
            "timesteps_shift": 1.0,
            "cfg_schedule": "constant",
            "use_norm": False,
            "use_system_prompt": None,
            "system_prompt": None,
        },
    )


def run_once(omni: Omni, args: argparse.Namespace) -> float:
    """One end-to-end generate call. Returns wall-time in ms."""
    # Block on any pending async CUDA work *before* we start the clock so
    # queued kernels from the previous iteration don't leak into this
    # measurement. Mirrors the cudaDeviceSynchronize bracketing used by
    # RustInfer's pipeline-level timings.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = omni.generate(
        {"prompt": PROMPT, "negative_prompt": None},
        make_params(args),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Minimal sanity check that a real image came out — otherwise the
    # timing is meaningless.
    assert outputs, "omni.generate returned no outputs"
    first = outputs[0]
    assert first.request_output.images, "no images in request_output"
    return elapsed_ms


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f"vllm-omni Z-Image-Turbo bench")
    print(f"  model:  {args.model}")
    print(f"  size:   {args.width}x{args.height}")
    print(f"  steps:  {args.num_inference_steps}")
    print(f"  warmup: {args.warmup}   iters: {args.iters}")
    print("=" * 60)

    omni = build_omni(args.model)

    print(f"[warmup] running {args.warmup} iterations...")
    for i in range(args.warmup):
        ms = run_once(omni, args)
        print(f"  warmup {i}: {ms:.1f} ms")

    print(f"[bench] running {args.iters} iterations...")
    latencies: list[float] = []
    for i in range(args.iters):
        ms = run_once(omni, args)
        latencies.append(ms)
        print(f"  iter {i}: {ms:.1f} ms")

    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    print()
    print("=" * 60)
    print(f"RESULT ({args.width}x{args.height}, {args.num_inference_steps} steps)")
    print(f"  mean:   {mean:7.1f} ms")
    print(f"  median: {median:7.1f} ms")
    print(f"  stdev:  {stdev:7.1f} ms")
    print(f"  min:    {min(latencies):7.1f} ms")
    print(f"  max:    {max(latencies):7.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
