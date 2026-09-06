"""Export HF Qwen3.5 layer/logit references and compare Rust diagnostic dumps.

Run the ignored worker checkpoint test with QWEN35_INPUT_JSON=<output>/inputs.json
and QWEN35_DUMP_DIR=<rust-dir>, then rerun this script with --compare <rust-dir>.
Files contain flattened little-endian FP32 tensors; comparison uses fixed tokens.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference_dir, actual_dir):
    steps = json.loads((reference_dir / "inputs.json").read_text())
    metadata = json.loads((reference_dir / "metadata.json").read_text())
    names = ["embed"] + [f"layer{i}" for i in range(metadata["num_layers"])] + ["logits"]
    report = []
    for step in range(len(steps)):
        for name in names:
            filename = f"step{step}_{name}.f32"
            ref = np.fromfile(reference_dir / filename, dtype="<f4").astype(np.float64)
            got = np.fromfile(actual_dir / filename, dtype="<f4").astype(np.float64)
            if ref.shape != got.shape or not np.isfinite(got).all():
                raise ValueError(f"Invalid output: {filename}")
            row = {
                "step": step,
                "tensor": name,
                "max_abs": float(np.abs(ref - got).max()),
                "relative_l2": float(np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1e-30)),
                "cosine": float(np.dot(ref, got) / max(np.linalg.norm(ref) * np.linalg.norm(got), 1e-30)),
            }
            if name == "logits":
                row.update(reference_top1=int(ref.argmax()), actual_top1=int(got.argmax()))
                sampled_path = actual_dir / f"step{step}_sampled.json"
                if sampled_path.exists():
                    row["sampled_token"] = json.loads(sampled_path.read_text())[0]
                    if row["sampled_token"] != row["actual_top1"]:
                        raise ValueError(f"Device/host argmax mismatch at step {step}")
                print(json.dumps(row))
            report.append(row)
    (actual_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    logits = [row for row in report if row["tensor"] == "logits"]
    agreement = sum(row["reference_top1"] == row["actual_top1"] for row in logits)
    print(f"top1 agreement: {agreement}/{len(logits)}")


def export(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        input_ids = torch.tensor([ids], device=args.device)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=args.decode_tokens,
        )
    continuation = output[0, len(ids):].tolist()
    steps = [ids] + [[token] for token in continuation]
    (args.output / "inputs.json").write_text(json.dumps(steps))
    text = model.model.language_model
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attention": "eager",
        "num_layers": len(text.layers),
        "generated_ids": continuation,
        "generated_text": tokenizer.decode(continuation, skip_special_tokens=True),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    step = 0

    def dump(name, tensor):
        values = tensor.detach().float().cpu().numpy().astype("<f4")
        values.tofile(args.output / f"step{step}_{name}.f32")

    text.embed_tokens.register_forward_hook(lambda module, inputs, output: dump("embed", output))
    for i, layer in enumerate(text.layers):
        layer.register_forward_hook(
            lambda module, inputs, output, i=i: dump(
                f"layer{i}", output if torch.is_tensor(output) else output[0]
            )
        )
    cache = None
    with torch.inference_mode():
        for step, ids in enumerate(steps):
            result = model(
                input_ids=torch.tensor([ids], device=args.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = result.past_key_values
            dump("logits", result.logits[:, -1, :])
    print(f"reference dumps complete: {len(steps)} steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--decode-tokens", type=int, default=12)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(args.output, args.compare)
    else:
        export(args)


if __name__ == "__main__":
    main()
