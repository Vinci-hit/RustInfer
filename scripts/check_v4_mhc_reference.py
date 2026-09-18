"""Check CUDA mHC against Transformers 5.12.0, without downloading weights.

Pass a shared build of v4_mhc.cu via --library. Rust integration tests separately
cover the public FusedOps wrappers, tensor contracts and graph replay.
"""

import argparse
import ctypes as C
import json
import math
from pathlib import Path


def ptr(tensor):
    return tensor.data_ptr() if tensor is not None else None


def bf_close(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    assert (diff <= expected.float().abs() / 128 + 1e-5).all(), diff.max().item()
    return diff.max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
    )

    if transformers.__version__ != "5.12.0":
        parser.error("Use transformers==5.12.0 to keep the reference fixed")
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(args.library.resolve()))
    pre = lib.rustinfer_v4_mhc_pre_bf16
    pre.argtypes = (
        [C.c_void_p] * 8 + [C.c_int] * 3 + [C.c_float] * 2 + [C.c_int, C.c_void_p]
    )
    pre.restype = C.c_int
    post = lib.rustinfer_v4_mhc_post_bf16
    post.argtypes = [C.c_void_p] * 5 + [C.c_int] * 2 + [C.c_void_p]
    post.restype = C.c_int

    with torch.inference_mode():
        for n, d in [(1, 128), (137, 128), (5, 4096), (1, 8192)]:
            torch.manual_seed(47)
            cfg = DeepseekV4Config(
                hidden_size=d, hc_mult=4, hc_sinkhorn_iters=20,
                hc_eps=1e-6, rms_norm_eps=1e-6,
            )
            hc = DeepseekV4HyperConnection(cfg).cuda().eval()
            hd = DeepseekV4HyperHead(cfg).cuda().eval()
            hc.fn.copy_(torch.randn_like(hc.fn) / math.sqrt(4 * d))
            hc.scale.copy_(torch.tensor([0.7, -1.3, 2.1], device="cuda"))
            hc.base.copy_(torch.randn_like(hc.base))
            hd.hc_fn.copy_(hc.fn[:4])
            hd.hc_scale.copy_(hc.scale[:1])
            hd.hc_base.copy_(hc.base[:4])
            x = torch.randn(1, n, 4, d, device="cuda", dtype=torch.bfloat16)
            branch = torch.randn(1, n, d, device="cuda", dtype=torch.bfloat16)
            y = torch.empty_like(branch)
            p = torch.empty(1, n, 4, device="cuda")
            c = torch.empty(1, n, 4, 4, device="cuda")
            out = torch.empty_like(x)
            head = torch.empty_like(branch)
            scratch = torch.empty(n * ((4 * d + 255) // 256) * 25, device="cuda")
            stream = torch.cuda.current_stream().cuda_stream
            assert pre(
                *map(ptr, [x, hc.fn, hc.scale, hc.base, scratch, y, p, c]),
                n, d, 0, 1e-6, 1e-6, 20, stream,
            ) == 0
            assert post(*map(ptr, [x, branch, p, c, out]), n, d, stream) == 0
            assert pre(
                *map(ptr, [out, hd.hc_fn, hd.hc_scale, hd.hc_base, scratch, head, None, None]),
                n, d, 1, 1e-6, 1e-6, 1, stream,
            ) == 0
            rp, rc, ry = hc(x)
            # Isolate Post from mapping roundoff by using the kernel's p/c.
            ro = p.to(x.dtype).unsqueeze(-1) * branch.unsqueeze(-2) + torch.matmul(
                c.to(x.dtype).transpose(-1, -2), x,
            )
            rh = hd(out)
            torch.cuda.synchronize()
            torch.testing.assert_close(p, rp, atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(c, rc, atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(out, ro, atol=0, rtol=0)
            print(json.dumps({
                "torch": torch.__version__, "transformers": transformers.__version__,
                "tokens": n, "dim": d,
                "pre_max_abs": bf_close(y, ry),
                "post_max_abs": (p - rp).abs().max().item(),
                "comb_max_abs": (c - rc).abs().max().item(),
                "expand_bitwise_equal": True,
                "head_max_abs": bf_close(head, rh),
            }))


if __name__ == "__main__":
    main()
