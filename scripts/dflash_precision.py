#!/usr/bin/env python3
"""Generate DFlash fixtures using the checkpoint author's original Python model."""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM


def author_model(path):
    spec = importlib.util.spec_from_file_location("dflash_author", path / "modeling_dflash.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DFlashDraftModel


def run_draft(head, target, features, anchor, block_size):
    device = features.device
    ids = torch.full((1, block_size), head.config.dflash_config["mask_token_id"], device=device, dtype=torch.long)
    ids[:, 0] = anchor
    cache = DynamicCache()
    n = features.shape[1]
    hidden = head(target_hidden=features, noise_embedding=target.get_input_embeddings()(ids),
                  position_ids=torch.arange(n + block_size, device=device)[None],
                  past_key_values=cache, use_cache=True, is_causal=False)
    logits = target.lm_head(hidden[:, 1:])
    return logits[0], cache


def generate_tiny(args):
    out = args.tiny_output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(473)
    tc = Qwen3Config(hidden_size=8, intermediate_size=16, num_hidden_layers=6,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=16,
                     max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-6,
                     tie_word_embeddings=False, attention_bias=False)
    tc._attn_implementation = "eager"
    target = Qwen3ForCausalLM(tc).eval()
    target.load_state_dict(load_file(str(Path(__file__).resolve().parents[1] / "crates/infer-worker/tests/fixtures/eagle3/target.safetensors")))
    cfg = dict(architectures=["DFlashDraftModel"], model_type="qwen3", hidden_size=8,
               intermediate_size=16, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
               vocab_size=16, num_hidden_layers=3, num_target_layers=6, max_position_embeddings=64,
               block_size=8, rms_norm_eps=1e-6, rope_theta=7000.0, dtype="bfloat16",
               hidden_act="silu", attention_bias=False, tie_word_embeddings=True,
               layer_types=["full_attention"] * 3,
               dflash_config=dict(mask_token_id=15, target_layer_ids=[1, 2, 3]))
    dc = Qwen3Config(**cfg)
    dc._attn_implementation = "eager"
    draft = author_model(args.draft)(dc).eval()
    with torch.no_grad():
        for p in draft.parameters():
            p.copy_(torch.ones_like(p) if p.ndim == 1 else torch.randn_like(p) * .12)
    draft.bfloat16()
    save_file(draft.state_dict(), str(out / "draft.safetensors"))
    draft.float()
    # Rebuild the nonpersistent frequency buffer after dtype conversion; model
    # parameters are BF16-roundtripped, RoPE frequencies remain FP32.
    draft.rotary_emb = type(draft.rotary_emb)(dc)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    prompt = [1, 4, 3, 7, 2]
    trajectory = list(prompt)
    with torch.inference_mode():
        for _ in range(24):
            trajectory.append(int(target(torch.tensor([trajectory])).logits[0, -1].argmax()))
        observed = target(torch.tensor([trajectory]), output_hidden_states=True)
        features = torch.cat([observed.hidden_states[i + 1] for i in draft.target_layer_ids], -1)
        logits, cache = run_draft(draft, target, features[:, :5], trajectory[5], 8)
        golden = dict(prompt=prompt, trajectory=trajectory, features=features[0].tolist(), logits=logits.tolist(),
                      target_logits=observed.logits[0].tolist(),
                      context_k=[cache[i][0][0, :, :5].transpose(0, 1).reshape(5, 4).tolist() for i in range(3)],
                      context_v=[cache[i][1][0, :, :5].transpose(0, 1).reshape(5, 4).tolist() for i in range(3)])
    golden["reference"] = dict(torch=torch.__version__, source=str(args.draft / "modeling_dflash.py"),
                                sha256=hashlib.sha256((args.draft / "modeling_dflash.py").read_bytes()).hexdigest())
    (out / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")


def generate_checkpoint(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    text = "<|im_start|>user\nExplain how hash tables work.\n<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
             attn_implementation="eager", local_files_only=True).cuda().eval()
    cfg = Qwen3Config.from_pretrained(args.draft, local_files_only=True)
    cfg._attn_implementation = "eager"
    draft = author_model(args.draft)(cfg).bfloat16().eval()
    draft.load_state_dict(load_file(str(args.draft / "model.safetensors")))
    draft.rotary_emb = type(draft.rotary_emb)(cfg)
    draft.cuda()
    def save(name, tensor):
        (out / (name + ".f32")).write_bytes(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    with torch.inference_mode():
        observed = target(ids, output_hidden_states=True)
        features = torch.cat([observed.hidden_states[i + 1] for i in draft.target_layer_ids], -1)
        anchor = int(observed.logits[0, -1].argmax())
        logits, cache = run_draft(draft, target, features, anchor, cfg.block_size)
        save("features", features)
        save("logits", logits)
        for i in range(cfg.num_hidden_layers):
            for j, label in enumerate(("k", "v")):
                save(f"context_{label}_{i}", cache[i][j][0, :, :ids.shape[1]].transpose(0, 1))
    (out / "metadata.json").write_text(json.dumps(dict(target=str(args.target), draft=str(args.draft),
        tokens=ids[0].tolist(), anchor=anchor, block_size=cfg.block_size, torch=torch.__version__,
        source_sha256=hashlib.sha256((args.draft / "modeling_dflash.py").read_bytes()).hexdigest()), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--target", type=Path)
    p.add_argument("--tiny-output", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.tiny_output:
        generate_tiny(args)
    elif args.output and args.target:
        generate_checkpoint(args)
    else:
        p.error("use --tiny-output, or --target and --output")
