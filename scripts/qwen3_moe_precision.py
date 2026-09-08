"""Export fixed-token HF Qwen3 MoE references or compare Rust diagnostic dumps."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare", type=Path)
parser.add_argument("--suite", action="store_true", help="Export a diverse 8-prompt, 16-step greedy reference suite")
parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
steps = [] if args.suite else json.loads(args.inputs.read_text())
args.output.mkdir(parents=True, exist_ok=True)
if args.compare:
    report = []
    for step, ids in enumerate(steps):
        if not ids:
            continue
        ref = np.fromfile(args.output / f"step{step}_logits.f32", dtype="<f4").astype(float)
        got = np.fromfile(args.compare / f"step{step}_logits.f32", dtype="<f4").astype(float)
        assert ref.shape == got.shape and np.isfinite(got).all()
        row = dict(step=step, relative_l2=float(np.linalg.norm(ref-got)/np.linalg.norm(ref)),
                   max_abs=float(abs(ref-got).max()), reference_top1=int(ref.argmax()),
                   actual_top1=int(got.argmax()))
        order = np.argsort(-ref, kind="stable")
        actual = row["actual_top1"]
        row.update(reference_margin=float(ref[order[0]]-ref[order[1]]),
                   actual_reference_gap=float(ref[order[0]]-ref[actual]),
                   actual_reference_rank=int(np.flatnonzero(order == actual)[0]+1),
                   reference_top5=order[:5].tolist())
        logp = ref - ref.max(); logp -= np.log(np.exp(logp).sum())
        logq = got - got.max(); logq -= np.log(np.exp(logq).sum())
        row["kl_ref_actual"] = float(np.sum(np.exp(logp)*(logp-logq)))
        print(json.dumps(row))
        report.append(row)
    (args.compare / "hf-comparison.json").write_text(json.dumps(report, indent=2))
    mismatches = [row for row in report if row["reference_top1"] != row["actual_top1"]]
    summary = dict(steps=len(report), top1_matches=len(report)-len(mismatches),
                   mismatches=mismatches,
                   max_actual_reference_rank=max(row["actual_reference_rank"] for row in report),
                   metrics={name: dict(zip(["median", "p95", "max"],
                                          np.quantile([row[name] for row in report], [.5, .95, 1]).tolist()))
                            for name in ["relative_l2", "kl_ref_actual", "max_abs"]})
    (args.compare / "hf-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
else:
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    import transformers
    (args.output / "environment.json").write_text(json.dumps(dict(
        torch=torch.__version__, transformers=transformers.__version__,
        model=str(args.model), attention=args.attention, dtype="bfloat16", device=args.device,
    ), indent=2))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attention
    ).to(args.device).eval()
    def save(name, tensor):
        tensor.detach().float().cpu().numpy().tofile(args.output / f"step{step}_{name}.f32")
    hooks = [layer.register_forward_hook(
        lambda module, inputs, output, i=i: save(f"layer{i}", output[0] if isinstance(output, tuple) else output)
    ) for i, layer in enumerate(model.model.layers)]
    cache = None
    if args.suite:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = [
            "The capital of France is",
            "Compute 17 + 25. Answer:",
            "Translate to English: 今天天气很好。 Translation:",
            "def factorial(n):\n    if n == 0:\n        return 1\n    return",
            "The three primary colors of light are",
            "请用一句话解释什么是机器学习：",
            "Extract the name and age as JSON: Alice is 30 years old. JSON:",
            "Alice has a red book. Bob has a blue pen. " * 8 + "What color is Bob's pen? Answer:",
        ]
        metadata = []
        with torch.inference_mode():
            for prompt in prompts:
                if steps:
                    steps.append([])
                cache = None
                ids = tokenizer.encode(prompt)
                first_step = len(steps)
                generated = []
                for _ in range(16):
                    step = len(steps)
                    steps.append(ids)
                    output = model(input_ids=torch.tensor([ids], device=args.device), past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    save("logits", output.logits[:, -1])
                    ids = [int(output.logits[0, -1].argmax())]
                    generated += ids
                metadata.append(dict(prompt=prompt, first_step=first_step, generated=generated,
                                     text=tokenizer.decode(generated, skip_special_tokens=True)))
                print("exported:", prompt[:60], flush=True)
        args.inputs.write_text(json.dumps(steps))
        (args.output / "suite.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        for hook in hooks:
            hook.remove()
        raise SystemExit(0)
    with torch.inference_mode():
        for step, ids in enumerate(steps):
            if not ids:
                cache = None
                continue
            output = model(input_ids=torch.tensor([ids], device=args.device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            save("logits", output.logits[:, -1])
            print("exported step", step, flush=True)
    for hook in hooks:
        hook.remove()
