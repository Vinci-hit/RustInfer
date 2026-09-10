"""Qwen3.5 MTP numerical reference using official Transformers layer operations.

Transformers does not load MTP natively. This explicitly builds the head from
checkpoint tensors following vLLM's qwen3_5_mtp.py: dual norm, [embedding, hidden]
concat, fc, one full-attention decoder, final norm, shared LM head.

Export, run worker test qwen35_mtp_checkpoint with QWEN35_MTP_REFERENCE and
QWEN35_MTP_DUMP_DIR, then --compare. Files are flattened little-endian FP32.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np


def compare(args):
    meta = json.loads((args.output / "metadata.json").read_text())
    report = []
    for name in ["target", "isolated_hidden", "isolated_logits", "integrated_hidden", "integrated_logits", "chunked_hidden", "chunked_logits"]:
        source = "target" if name == "target" else "mtp_" + name.rsplit("_", 1)[1]
        ref = np.fromfile(args.output / (source + ".f32"), dtype="<f4").astype(np.float64)
        got = np.fromfile(args.compare / (name + ".f32"), dtype="<f4").astype(np.float64)
        assert ref.shape == got.shape and ref.size and np.isfinite(ref).all() and np.isfinite(got).all(), name
        error = float(np.linalg.norm(ref-got) / max(np.linalg.norm(ref), 1e-30))
        row = dict(tensor=name, relative_l2=error, max_abs=float(np.max(np.abs(ref-got))))
        limit = 0.03 if name.startswith(("isolated", "chunked")) else 0.08
        assert error < limit, row
        if name.endswith("logits"):
            r = ref.reshape(-1, meta["vocab_size"])
            g = got.reshape(r.shape)
            agree = r.argmax(-1) == g.argmax(-1)
            row.update(top1_matches=int(agree.sum()), rows=len(agree), last_top1_match=bool(agree[-1]))
            # BF16 can reverse near ties; large-margin decisions must agree.
            top = np.partition(r, -2, axis=1)[:, -2:]
            confident = top[:,1] - top[:,0] > 0.5
            assert agree[confident].all() and agree[-1], row
        report.append(row)
        print(json.dumps(row))
    # Chunk equivalence uses the identical conditioning hidden, isolating cache alignment.
    for suffix in ["hidden", "logits"]:
        full = np.fromfile(args.compare / f"isolated_{suffix}.f32", dtype="<f4").astype(np.float64)
        chunk = np.fromfile(args.compare / f"chunked_{suffix}.f32", dtype="<f4").astype(np.float64)
        error = float(np.linalg.norm(full-chunk) / max(np.linalg.norm(full), 1e-30))
        assert error < 0.02, (suffix,error)
        report.append(dict(tensor="whole_vs_chunked_"+suffix, relative_l2=error))
    (args.compare / "comparison.json").write_text(json.dumps(report,indent=2))


def export(args):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").to(args.device).eval()
    text = model.model.language_model
    cfg = copy.deepcopy(model.config.text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    cfg._attn_implementation = "eager"
    weights = {}
    for file in Path(args.model).glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith("mtp."):
                    weights[name[4:]] = f.get_tensor(name)
    assert len(weights) == 15, sorted(weights)
    def norm(name):
        layer = Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        layer.load_state_dict({"weight": weights[name+".weight"]})
        return layer.to(device=args.device, dtype=torch.bfloat16).eval()
    en, hn, final = norm("pre_fc_norm_embedding"), norm("pre_fc_norm_hidden"), norm("norm")
    fc = torch.nn.Linear(2*cfg.hidden_size, cfg.hidden_size, bias=False)
    fc.load_state_dict({"weight":weights["fc.weight"]})
    fc = fc.to(device=args.device,dtype=torch.bfloat16).eval()
    block = Qwen3_5DecoderLayer(cfg,0)
    block.load_state_dict({k[len("layers.0."):]:v for k,v in weights.items() if k.startswith("layers.0.")})
    block = block.to(device=args.device,dtype=torch.bfloat16).eval()
    rope = Qwen3_5TextRotaryEmbedding(cfg,device=args.device)
    ids = tokenizer.encode(args.prompt,add_special_tokens=False)
    args.output.mkdir(parents=True,exist_ok=True)
    def dump(name,t):
        t.detach().float().cpu().numpy().astype("<f4").tofile(args.output / (name+".f32"))
    with torch.inference_mode():
        tokens = torch.tensor([ids],device=args.device)
        target = text(input_ids=tokens,use_cache=False).last_hidden_state
        dump("target",target)
        n = len(ids)-1
        x = fc(torch.cat([en(text.embed_tokens(tokens[:,1:])), hn(target[:,:-1])],dim=-1))
        positions = torch.arange(n,device=args.device)[None,:]
        mask = torch.full((n,n), torch.finfo(x.dtype).min, device=args.device,dtype=x.dtype).triu(1)[None,None]
        out = final(block(x,position_embeddings=rope(x,positions),attention_mask=mask))
        logits = model.lm_head(out)
        dump("mtp_hidden",out)
        dump("mtp_logits",logits)
    (args.output / "tokens.json").write_text(json.dumps(ids))
    meta = dict(model=args.model, prompt=args.prompt,tokens=len(ids),hidden_size=cfg.hidden_size,vocab_size=cfg.vocab_size,
                torch=torch.__version__,transformers=transformers.__version__,dtype="bfloat16",
                reference="Transformers Qwen3.5 layer operations + vLLM MTP head composition",
                sources=["https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py",
                         "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py"])
    (args.output / "metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/md2/liuwenqi/vllm_bench/Qwen3.5-4B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path,required=True)
    p.add_argument("--compare", type=Path)
    p.add_argument("--prompt", default="The capital of France is Paris. 2 + 2 = 4. 请用中文解释 Rust 的所有权：每个值都有一个所有者。 Python example: def add(a, b): return a + b. The next prime after 7 is")
    args = p.parse_args()
    compare(args) if args.compare else export(args)
