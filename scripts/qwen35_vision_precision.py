"""Export reproducible image preprocessing, ViT, and multimodal decoder references."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--processor-only", action="store_true")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        y, x = np.indices((270, 350))
        rgb = np.stack([(x * 3) % 256, (y * 5) % 256, ((x // 45 + y // 45) % 2) * 255], -1).astype(np.uint8)
        image = Image.fromarray(rgb)
    image.save(args.output / "image.png")
    image.save(args.output / "image.jpg", quality=95)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size["longest_edge"] = 512 * 32 * 32
    cases = []
    for j, (height, width) in enumerate([(31, 47), (768, 1024), (900, 100), (64, 64), (270, 350)]):
        rng = np.random.default_rng(j)
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        rgb.tofile(args.output / f"processor{j}.rgb")
        result = processor.image_processor(images=Image.fromarray(rgb), return_tensors="pt")
        result["pixel_values"].bfloat16().view(torch.uint16).numpy().astype("<u2").tofile(args.output / f"processor{j}.bf16")
        cases.append({"height": height, "width": width, "grid": result["image_grid_thw"][0].tolist()})
    (args.output / "processor_cases.json").write_text(json.dumps(cases))
    if args.processor_only:
        print(json.dumps(cases))
        return
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                 {"type": "text", "text": "Describe the image briefly."}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    inputs["pixel_values"].float().numpy().astype("<f4").tofile(args.output / "pixels.f32")
    pixels = inputs["pixel_values"].to(torch.bfloat16)
    pixels.view(torch.uint16).numpy().astype("<u2").tofile(args.output / "patches.bf16")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                    attn_implementation=args.attn_implementation).to(args.device).eval()
    def dump(name, value):
        value.detach().float().cpu().numpy().astype("<f4").tofile(args.output / (name + ".f32"))
    visual = model.model.visual
    visual.patch_embed.register_forward_hook(lambda m, i, o: dump("vision_patch", o))
    for j, block in enumerate(visual.blocks):
        block.register_forward_hook(lambda m, i, o, j=j: dump(f"vision_block{j}", o))
    visual.blocks[0].register_forward_pre_hook(lambda m, i: dump("vision_position", i[0]))
    visual.merger.register_forward_hook(lambda m, i, o: dump("vision_merger", o))
    inputs = inputs.to(args.device)
    generated = []
    with torch.inference_mode():
        positions, delta = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                     image_grid_thw=inputs["image_grid_thw"])
        result = model(**inputs, use_cache=True)
        dump("step0_logits", result.logits[:, -1, :])
        for j in range(args.decode_tokens):
            token = int(result.logits[0, -1].argmax())
            generated.append(token)
            result = model(input_ids=torch.tensor([[token]], device=args.device),
                           past_key_values=result.past_key_values, use_cache=True)
            dump(f"step{j+1}_logits", result.logits[:, -1, :])
    metadata = {"input_ids": ids, "grid_thw": inputs["image_grid_thw"].cpu().tolist()[0],
                "positions": positions[:, 0].T.cpu().tolist(), "rope_delta": int(delta.item()),
                "generated_ids": generated, "text": processor.tokenizer.decode(generated),
                "prompt_text": "Describe the image briefly."}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"prompt_tokens":len(ids), "grid":metadata["grid_thw"], "generated":generated}))


if __name__ == "__main__":
    main()
