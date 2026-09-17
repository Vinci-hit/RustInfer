"""Create an offline, random-weight V4 fixture and independent Transformers outputs.

This is an architecture test, not a language model checkpoint. No Hub downloads.
Reference: transformers==5.12.0, eager attention, unquantized weights/cache.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=["micro", "tiny"], default="tiny")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, DynamicCache

    if transformers.__version__ != "5.12.0":
        parser.error("Use transformers==5.12.0 to keep the reference implementation fixed")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    micro = args.preset == "micro"
    config = DeepseekV4Config(
        vocab_size=64 if micro else 256,
        hidden_size=32 if micro else 128,
        moe_intermediate_size=16 if micro else 64,
        num_hidden_layers=4,
        num_attention_heads=2 if micro else 4,
        num_key_value_heads=1,
        head_dim=16 if micro else 32,
        q_lora_rank=16 if micro else 64,
        o_groups=2,
        o_lora_rank=8 if micro else 32,
        index_n_heads=2,
        index_head_dim=8 if micro else 16,
        index_topk=2,
        partial_rotary_factor=0.25,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=3,
        compress_ratios=[0, 4, 128, 4],
        sliding_window=16,
        max_position_embeddings=256,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        num_nextn_predict_layers=0,
        # Short-context fixture: plain RoPE on both branches, with different bases.
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config.rustinfer_tiny = True
    config.dtype = args.dtype
    dtype = getattr(torch, args.dtype)
    model = DeepseekV4ForCausalLM(config).eval()
    with torch.no_grad():
        # Keep indexer dot products away from the ReLU zero plateau. Otherwise
        # equal top-k scores permit different (equally valid) selections when
        # a full prefill has more future, masked entries than a decode call.
        model.model.embed_tokens.weight.uniform_(0.01, 0.04)
        # HF initializes this table to zero. Populate distinct routes so the
        # fixture exercises multiple experts rather than expert zero twice.
        for i, layer in enumerate(model.model.layers):
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                ids = torch.arange(config.vocab_size)[:, None]
                gate.tid2eid.copy_((ids + i + torch.arange(2)) % config.n_routed_experts)
            else:
                # Well-separated selection scores keep BF16 arithmetic drift
                # from changing the discrete expert set in a parity fixture.
                # Hash layers still exercise every expert; route weights remain
                # input-dependent and must exclude this correction bias.
                gate.e_score_correction_bias.copy_(torch.tensor([-0.9, 0.3, 0.9, -0.3]))
            # Exercise non-symmetric HC mixing and nonzero position biases/sinks.
            layer.attn_hc.base.normal_(std=0.15)
            layer.ffn_hc.base.normal_(std=0.15)
            layer.self_attn.sinks.normal_(std=0.1)
            compressor = layer.self_attn.compressor
            if compressor is not None:
                compressor.position_bias.normal_(std=0.1)
                if hasattr(compressor, "indexer"):
                    compressor.indexer.position_bias.normal_(std=0.1)
                    layer.self_attn.q_a_proj.weight.uniform_(0.005, 0.04)
                    compressor.indexer.kv_proj.weight.uniform_(0.005, 0.04)
                    compressor.indexer.q_b_proj.weight.uniform_(0.005, 0.04)
                    compressor.indexer.scorer.weights_proj.weight.uniform_(0.005, 0.04)
    model.to(device=args.device, dtype=dtype)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Cross CSA boundaries, sliding eviction and the first HCA emission.
    inputs = [int((i * 17 + i // 7 + 3) % config.vocab_size) for i in range(137)]
    cases = {"full": [137], "chunked": [3, 1, 11, 1, 111, 1, 1, 8], "decode": [127] + [1] * 10}
    reference = {}
    with torch.inference_mode():
        for case, chunks in cases.items():
            layer_parts = [[] for _ in model.model.layers]
            hooks = [
                layer.register_forward_hook(
                    lambda module, inp, out, i=i: layer_parts[i].append(
                        out.detach().float().cpu().reshape(-1, config.hc_mult * config.hidden_size)
                    )
                )
                for i, layer in enumerate(model.model.layers)
            ]
            cache = DynamicCache(config=config)
            logits, cursor = [], 0
            for length in chunks:
                ids = torch.tensor([inputs[cursor:cursor + length]], device=args.device)
                result = model(input_ids=ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits.append(result.logits[0].float().cpu())
                cursor += length
            for hook in hooks:
                hook.remove()
            reference[f"{case}.logits"] = torch.cat(logits)
            for i, parts in enumerate(layer_parts):
                reference[f"{case}.layer{i}"] = torch.cat(parts)
    if not all(torch.isfinite(v).all() for v in reference.values()):
        raise RuntimeError("reference produced non-finite values")
    for case in ("chunked", "decode"):
        atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (0.025, 0.03)
        torch.testing.assert_close(reference[f"{case}.logits"], reference["full.logits"], atol=atol, rtol=rtol)
    args.output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(args.output)
    weights = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, args.output / "model.safetensors")
    save_file(reference, args.output / "reference.safetensors")
    manifest = {
        "format": "rustinfer-v4-tiny-v1",
        "preset": args.preset,
        "seed": args.seed,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "reference_device": args.device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_bytes": sum(t.numel() * t.element_size() for t in weights.values()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "inputs": inputs,
        "cases": cases,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("inputs", "cases")}, indent=2))


if __name__ == "__main__":
    main()
