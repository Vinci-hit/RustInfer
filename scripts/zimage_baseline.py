"""
Z-Image-Turbo Python baseline (diffusers).

Generates a 1024x1024 image with 9 inference steps and saves to
/tmp/zimage_python.png. Acts as the visual reference for the
RustInfer port.

Usage:
    /root/test_env/.venv/bin/python /root/RustInfer/scripts/zimage_baseline.py
"""
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

MODEL_DIR = "/apdcephfs_qy2/share_303432435/vinciiliu/models/z-image-turbo"
OUT_PATH = "/tmp/zimage_python.png"
PROMPT = (
    "A majestic snow leopard standing on a cliff edge at sunset, "
    "with golden light illuminating its fur, dramatic mountain landscape "
    "in the background, photorealistic, 8k detail"
)


def main():
    if not Path(MODEL_DIR).is_dir():
        sys.exit(f"model dir not found: {MODEL_DIR}")

    print(f"loading pipeline from {MODEL_DIR}...", flush=True)
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    print("generating 1024x1024 / 9 steps / seed=42 ...", flush=True)
    t0 = time.time()
    gen = torch.Generator(device="cuda").manual_seed(42)
    out = pipe(
        prompt=PROMPT,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=gen,
    )
    img = out.images[0]
    print(f"  inference {time.time() - t0:.1f}s", flush=True)

    img.save(OUT_PATH)
    print(f"saved {OUT_PATH} (size={img.size}, mode={img.mode})", flush=True)


if __name__ == "__main__":
    main()
