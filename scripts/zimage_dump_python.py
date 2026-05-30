"""
Dump intermediate activations of `noise_refiner.0` for numerical comparison.
Saves f32 .npy to /tmp/zimage_dump_python/ (overwrite).
"""
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

MODEL_DIR = "/apdcephfs_qy2/share_303432435/vinciiliu/models/z-image-turbo"
OUT_DIR = Path("/tmp/zimage_dump_python")
PROMPT = (
    "A majestic snow leopard standing on a cliff edge at sunset, with golden "
    "light illuminating its fur, dramatic mountain landscape in the background, "
    "photorealistic, 8k detail"
)


def save_t(name, t):
    arr = t.detach().to(torch.float32).cpu().contiguous().numpy()
    np.save(OUT_DIR / f"{name}.npy", arr, allow_pickle=False)


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    pipe = DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    block = pipe.transformer.noise_refiner[0]
    saved_step = {"i": 0}
    SAVE_STEP = 0

    # Hook all_x_embedder["2-1"] for patches/x_emb.
    x_emb_module = pipe.transformer.all_x_embedder["2-1"]
    saved_x_emb = {"hooked": False}
    def x_emb_pre_hook(module, inputs):
        if saved_step["i"] == SAVE_STEP and not saved_x_emb["hooked"]:
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            save_t("step0_patches", x)
    def x_emb_post_hook(module, inputs, output):
        if saved_step["i"] == SAVE_STEP and not saved_x_emb["hooked"]:
            save_t("step0_x_emb", output)
            saved_x_emb["hooked"] = True
    x_emb_module.register_forward_pre_hook(x_emb_pre_hook)
    x_emb_module.register_forward_hook(x_emb_post_hook)

    # Capture initial latent from transformer pre-hook.
    saved_init = {"done": False}
    def transformer_pre_hook(module, args, kwargs):
        if not saved_init["done"]:
            x = args[0] if args else kwargs.get("x")
            if isinstance(x, list) and len(x) > 0:
                t0 = x[0]
                if isinstance(t0, list):
                    t0 = t0[-1]
                save_t("step0_initial_latent", t0)
                saved_init["done"] = True
    pipe.transformer.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True)

    # Capture x_freqs_cis and x_pos_ids by wrapping _prepare_sequence
    # using its unbound class method (avoids recursion).
    saved_seq = {"x_done": False}
    cls = type(pipe.transformer)
    orig_prepare_seq = cls._prepare_sequence
    def wrap_ps(self_t, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
        out = orig_prepare_seq(self_t, feats, pos_ids, inner_pad_mask, pad_token,
                                noise_mask=noise_mask, device=device)
        if not saved_seq["x_done"]:
            save_t("step0_x_pos_ids_py", torch.cat(pos_ids, dim=0))
            saved_seq["x_done"] = True
        return out
    cls._prepare_sequence = wrap_ps

    def patched_forward(x, attn_mask, freqs_cis, adaln_input=None,
                        noise_mask=None, adaln_noisy=None, adaln_clean=None,
                        **kwargs):
        if saved_step["i"] != SAVE_STEP:
            return orig_block_forward(
                x, attn_mask, freqs_cis,
                adaln_input=adaln_input,
                noise_mask=noise_mask,
                adaln_noisy=adaln_noisy,
                adaln_clean=adaln_clean,
                **kwargs,
            )
        save_t("step0_x_padded_in", x)
        save_t("step0_adaln_input", adaln_input)
        save_t("step0_freqs_cis", torch.view_as_real(freqs_cis))

        mod = block.adaLN_modulation(adaln_input)
        save_t("step0_nr0_mod_out", mod)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        save_t("step0_nr0_scale_msa", scale_msa.squeeze(1))
        save_t("step0_nr0_gate_msa", gate_msa.squeeze(1))
        save_t("step0_nr0_scale_mlp", scale_mlp.squeeze(1))
        save_t("step0_nr0_gate_mlp", gate_mlp.squeeze(1))

        norm1_x = block.attention_norm1(x) * scale_msa
        save_t("step0_nr0_norm1_x", norm1_x)

        attn = block.attention
        q = attn.to_q(norm1_x); k = attn.to_k(norm1_x); v = attn.to_v(norm1_x)
        save_t("step0_nr0_qkv_q", q)
        save_t("step0_nr0_qkv_k", k)
        save_t("step0_nr0_qkv_v", v)

        q4 = q.unflatten(-1, (attn.heads, -1))
        k4 = k.unflatten(-1, (attn.heads, -1))
        v4 = v.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None: q4 = attn.norm_q(q4)
        if attn.norm_k is not None: k4 = attn.norm_k(k4)
        save_t("step0_nr0_q_normed", q4)
        save_t("step0_nr0_k_normed", k4)

        def apply_rotary_emb(x_in, fc):
            with torch.amp.autocast("cuda", enabled=False):
                xc = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                fc2 = fc.unsqueeze(2)
                x_out = torch.view_as_real(xc * fc2).flatten(3)
                return x_out.type_as(x_in)

        q_rope = apply_rotary_emb(q4, freqs_cis)
        k_rope = apply_rotary_emb(k4, freqs_cis)
        save_t("step0_nr0_q_roped", q_rope)
        save_t("step0_nr0_k_roped", k_rope)

        q_bhsd = q_rope.transpose(1, 2)
        k_bhsd = k_rope.transpose(1, 2)
        v_bhsd = v4.transpose(1, 2)
        sdpa_out = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=False)
        sdpa_out_shd = sdpa_out.transpose(1, 2)
        save_t("step0_nr0_attn_out_premerge", sdpa_out_shd)

        attn_flat = sdpa_out_shd.flatten(2, 3)
        attn_proj = attn.to_out[0](attn_flat)
        save_t("step0_nr0_attn_out_post", attn_proj)

        norm2_attn = gate_msa * block.attention_norm2(attn_proj)
        save_t("step0_nr0_norm2_attn", norm2_attn)
        x_after_attn = x + norm2_attn
        save_t("step0_nr0_after_attn", x_after_attn)

        norm1_ffn = block.ffn_norm1(x_after_attn) * scale_mlp
        save_t("step0_nr0_norm1_ffn", norm1_ffn)

        ff = block.feed_forward
        w1_out = ff.w1(norm1_ffn); w3_out = ff.w3(norm1_ffn)
        save_t("step0_nr0_w1_out", w1_out)
        save_t("step0_nr0_w3_out", w3_out)
        silu_w1 = F.silu(w1_out) * w3_out
        save_t("step0_nr0_silu_w1_x_w3", silu_w1)
        w2_out = ff.w2(silu_w1)
        save_t("step0_nr0_w2_out", w2_out)
        norm2_ffn = gate_mlp * block.ffn_norm2(w2_out)
        save_t("step0_nr0_norm2_ffn", norm2_ffn)
        block_out = x_after_attn + norm2_ffn
        save_t("step0_nr0_block_out", block_out)
        return block_out

    orig_block_forward = block.forward
    block.forward = patched_forward

    orig_step = pipe.scheduler.step
    def counting_step(*args, **kw):
        out = orig_step(*args, **kw)
        saved_step["i"] += 1
        return out
    pipe.scheduler.step = counting_step

    gen = torch.Generator(device="cuda").manual_seed(42)
    pipe(prompt=PROMPT, height=1024, width=1024, num_inference_steps=9,
         guidance_scale=0.0, generator=gen)

    n = len(list(OUT_DIR.glob("*.npy")))
    print(f"saved {n} npy files to {OUT_DIR}")


if __name__ == "__main__":
    main()
