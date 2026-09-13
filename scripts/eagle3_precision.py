#!/usr/bin/env python3
"""Generate independent PyTorch fixtures for Qwen3 EAGLE3 components.

The draft equations follow DeepSpec 005e03b's Qwen3Eagle3Model. The target
oracle is Transformers Qwen3. CPU fixtures require no GPU at test time.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def draft_forward(w, cfg, features, ids, positions, cache=None):
    # Canonicalize the two published naming conventions for this independent
    # equation-level reference (Rust's format dispatch is not used here).
    w = {k.replace("midlayer.", "layers.0."): v for k, v in w.items()}
    h = cfg["hidden_size"]
    nh, nk, hd = (cfg[k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    eps = cfg["rms_norm_eps"]

    def norm(x, name):
        f = x.float()
        return (f * (f.square().mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * w[name]

    hidden = F.linear(features, w["fc.weight"]) if features.shape[-1] != h else features
    inputs = torch.cat((norm(F.embedding(ids, w["embed_tokens.weight"]), "layers.0.input_layernorm.weight"),
                        norm(hidden, "layers.0.hidden_norm.weight")), -1)
    q, k, v = [F.linear(inputs, w[f"layers.0.self_attn.{name}_proj.weight"]).view(len(ids), heads, hd).transpose(0, 1)
               for name, heads in (("q", nh), ("k", nk), ("v", nk))]
    if "layers.0.self_attn.q_norm.weight" in w:
        q = norm(q, "layers.0.self_attn.q_norm.weight")
        k = norm(k, "layers.0.self_attn.k_norm.weight")
    theta = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta"))
    inv = theta ** (-torch.arange(0, hd, 2, device=positions.device).float() / hd)
    angles = torch.outer(positions.float(), inv).repeat(1, 2)[None]

    def rope(x):
        rotated = torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), -1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    q, k = rope(q), rope(k)
    if cache is not None:
        k, v = torch.cat((cache[0], k), 1), torch.cat((cache[1], v), 1)
    mask = torch.arange(k.shape[1], device=positions.device)[None, :] <= positions[:, None]
    attn = F.scaled_dot_product_attention(q, k.repeat_interleave(nh // nk, 0),
                                         v.repeat_interleave(nh // nk, 0), attn_mask=mask)
    residual = hidden + F.linear(attn.transpose(0, 1).reshape(len(ids), nh * hd), w["layers.0.self_attn.o_proj.weight"])
    inputs = norm(residual, "layers.0.post_attention_layernorm.weight")
    mlp = F.silu(F.linear(inputs, w["layers.0.mlp.gate_proj.weight"])) * F.linear(inputs, w["layers.0.mlp.up_proj.weight"])
    hidden = residual + F.linear(mlp, w["layers.0.mlp.down_proj.weight"])
    logits = F.linear(norm(hidden, "norm.weight"), w["lm_head.weight"])
    return hidden, logits, (k, v)


def generate_tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7319)
    cfg = Qwen3Config(hidden_size=8, intermediate_size=16, num_hidden_layers=6,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                      vocab_size=16, max_position_embeddings=64, rope_theta=10000.0,
                      rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=False)
    cfg._attn_implementation = "eager"
    target = Qwen3ForCausalLM(cfg).eval().to(torch.bfloat16)
    save_file(target.state_dict(), str(output / "target.safetensors"))
    target = target.float()
    draft_cfg = dict(architectures=["Qwen3Eagle3Model"], hidden_size=8, intermediate_size=16,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=16,
                     num_hidden_layers=1, draft_num_hidden_layers=1, num_target_layers=6,
                     target_layer_ids=[0, 1, 2, 3, 4], target_model_name_or_path="test/Qwen3-tiny",
                     max_position_embeddings=64, rms_norm_eps=1e-6,
                     rope_parameters=dict(rope_theta=10000.0, rope_type="default"),
                     dtype="bfloat16", hidden_act="silu", attention_bias=False,
                     tie_word_embeddings=False, ttt_length=7)
    shapes = {"embed_tokens.weight": (16, 8), "lm_head.weight": (16, 8), "fc.weight": (8, 40), "norm.weight": (8,)}
    shapes.update({f"layers.0.{n}.weight": (8,) for n in ("hidden_norm", "input_layernorm", "post_attention_layernorm")})
    shapes.update({f"layers.0.self_attn.{n}_norm.weight": (4,) for n in ("q", "k")})
    shapes.update({f"layers.0.self_attn.{n}_proj.weight": s for n, s in (("q", (8, 16)), ("k", (4, 16)), ("v", (4, 16)), ("o", (8, 8)))})
    shapes.update({f"layers.0.mlp.{n}_proj.weight": s for n, s in (("gate", (16, 8)), ("up", (16, 8)), ("down", (8, 16)))})
    weights = {name: (torch.ones(shape) if len(shape) == 1 else torch.randn(shape) * 0.12).to(torch.bfloat16) for name, shape in shapes.items()}
    save_file(weights, str(output / "draft.safetensors"))
    (output / "config.json").write_text(json.dumps(draft_cfg, indent=2) + "\n")
    weights = {k: v.float() for k, v in weights.items()}
    ids = [1, 4, 3, 7, 2]
    with torch.no_grad():
        # A complete greedy trajectory supplies true target features for cache
        # reconstruction tests after arbitrary speculative rejection lengths.
        trajectory = list(ids)
        for _ in range(20):
            trajectory.append(int(target(torch.tensor([trajectory])).logits[0, -1].argmax()))
        observed = target(torch.tensor([trajectory]), output_hidden_states=True)
        features = torch.cat([observed.hidden_states[i + 1] for i in draft_cfg["target_layer_ids"]], -1)[0]
        paired = torch.tensor(trajectory[1:6])
        hidden, logits, cache = draft_forward(weights, draft_cfg, features[:5], paired, torch.arange(5))
        next_token = logits[-1:].argmax(-1)
        next_hidden, next_logits, next_cache = draft_forward(weights, draft_cfg, hidden[-1:], next_token, torch.tensor([5]), cache)
    golden = dict(prompt=ids, trajectory=trajectory, features=features.tolist(), target_logits=observed.logits[0].tolist(),
                  paired=paired.tolist(), projected=F.linear(features[:5], weights["fc.weight"]).tolist(),
                  hidden=hidden.tolist(), logits=logits.tolist(), k=cache[0].transpose(0, 1).reshape(5, 4).tolist(),
                  v=cache[1].transpose(0, 1).reshape(5, 4).tolist(), next_token=next_token.tolist(),
                  next_hidden=next_hidden.tolist(), next_logits=next_logits.tolist(),
                  next_k=next_cache[0].transpose(0, 1).reshape(6, 4).tolist())
    (output / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    generate_specforge(output / "specforge", target, trajectory, draft_cfg)


def generate_specforge(output, target, trajectory, base_cfg):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(991)
    cfg = {k: v for k, v in base_cfg.items() if k not in
           ("target_layer_ids", "draft_num_hidden_layers", "num_target_layers", "target_model_name_or_path", "rope_parameters", "ttt_length")}
    cfg.update(architectures=["LlamaForCausalLMEagle3"], draft_vocab_size=8, mlp_bias=False,
               rope_theta=7000.0, eagle_aux_hidden_state_layer_ids=[0, 2, 4])
    names = {"fc.weight": (8, 24), "lm_head.weight": (8, 8), "norm.weight": (8,)}
    names.update({f"midlayer.{n}.weight": (8,) for n in ("hidden_norm", "input_layernorm", "post_attention_layernorm")})
    names.update({f"midlayer.self_attn.{n}_proj.weight": s for n, s in (("q", (8, 16)), ("k", (4, 16)), ("v", (4, 16)), ("o", (8, 8)))})
    names.update({f"midlayer.mlp.{n}_proj.weight": s for n, s in (("gate", (16, 8)), ("up", (16, 8)), ("down", (8, 16)))})
    weights = {name: (torch.ones(shape) if len(shape) == 1 else torch.randn(shape) * .12).to(torch.bfloat16) for name, shape in names.items()}
    mapping = torch.tensor([0, 2, 3, 5, 8, 11, 13, 15])
    weights["d2t"] = mapping - torch.arange(8)
    weights["t2d"] = torch.zeros(16, dtype=torch.bool)
    weights["t2d"][mapping] = True
    save_file(weights, str(output / "draft.safetensors"))
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    weights = {name: value.float() for name, value in weights.items()}
    weights["embed_tokens.weight"] = target.get_input_embeddings().weight.detach()
    with torch.no_grad():
        observed = target(torch.tensor([trajectory]), output_hidden_states=True)
        features = torch.cat([observed.hidden_states[i+1] for i in cfg["eagle_aux_hidden_state_layer_ids"]], -1)[0]
        paired = torch.tensor(trajectory[1:6])
        hidden, logits, cache = draft_forward(weights, cfg, features[:5], paired, torch.arange(5))
        next_id = mapping[logits[-1:].argmax(-1)]
        next_hidden, next_logits, next_cache = draft_forward(weights, cfg, hidden[-1:], next_id, torch.tensor([5]), cache)
        golden = dict(features=features.tolist(), paired=paired.tolist(), projected=F.linear(features[:5],weights["fc.weight"]).tolist(),
                      hidden=hidden.tolist(), logits=logits.tolist(), mapping=mapping.tolist(), next_token=next_id.tolist(),
                      next_hidden=next_hidden.tolist(), next_logits=next_logits.tolist(),
                      k=cache[0].transpose(0,1).reshape(5,4).tolist(), v=cache[1].transpose(0,1).reshape(5,4).tolist(),
                      next_k=next_cache[0].transpose(0,1).reshape(6,4).tolist())
    (output / "golden.json").write_text(json.dumps(golden, indent=2)+"\n")


def export_checkpoint(args):
    """Export BF16 target features and a real-checkpoint draft computation."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.draft / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
                                               attn_implementation="eager").to(args.device).eval()
    weights = load_file(str(args.draft / "model.safetensors"), device=args.device)
    if "embed_tokens.weight" not in weights:
        weights["embed_tokens.weight"] = model.get_input_embeddings().weight.detach()
    if cfg["architectures"] == ["LlamaForCausalLMEagle3"]:
        layers = cfg.get("eagle_config", {}).get("eagle_aux_hidden_state_layer_ids") or cfg.get("eagle_aux_hidden_state_layer_ids")
        layers = layers or [1, model.config.num_hidden_layers//2-1, model.config.num_hidden_layers-4]
    else:
        layers = cfg["target_layer_ids"]
    with torch.inference_mode():
        observed = model(torch.tensor([ids], device=args.device), output_hidden_states=True)
        features = torch.cat([observed.hidden_states[i + 1] for i in layers], -1)[0]
        n = len(ids) - 1
        projected = F.linear(features[:n], weights["fc.weight"])
        hidden, logits, _ = draft_forward(weights, cfg, features[:n], torch.tensor(ids[1:], device=args.device),
                                          torch.arange(n, device=args.device))
    for name, tensor in (("features", features), ("projected", projected), ("hidden", hidden), ("logits", logits)):
        tensor.float().cpu().numpy().astype("<f4").tofile(output / f"{name}.f32")
    (output / "metadata.json").write_text(json.dumps(dict(target=str(args.target), draft=str(args.draft), tokens=ids,
                                                       dim=cfg["hidden_size"], vocab=cfg["vocab_size"], draft_vocab=cfg.get("draft_vocab_size",cfg["vocab_size"]), layers=layers,
                                                       dtype="bfloat16", torch=torch.__version__), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="The capital of France is Paris. Please explain Rust ownership.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.target or args.draft:
        if not (args.target and args.draft):
            parser.error("--target and --draft are required together")
        export_checkpoint(args)
    else:
        generate_tiny(args.output)
