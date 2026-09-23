#!/usr/bin/env python3
"""Generate CPU-only text goldens using HF Tokenizers and Python Jinja2.

Inputs are the existing GGUF metadata inspection JSON and HF tokenizer.json.
No model weights or llama.cpp execution are involved. The canonical Qwen3.5
profile is explicit: local vendor tokenizer.json exports may contain a doubly
escaped Split regex or the older qwen2 pattern, so they are not used as-is.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jinja2
import tokenizers
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

PROFILE_SOURCE = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/tokenization_qwen3_5.py"
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--hf-tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    m = {v["key"]: v["value"] for v in manifest["metadata"]}
    hf = json.loads(args.hf_tokenizer.read_text())
    vocab = hf["model"]["vocab"]
    merges = [tuple(v.split(" ") if isinstance(v, str) else v) for v in hf["model"]["merges"]]
    assert all(m["tokenizer.ggml.tokens"][i] == t for t, i in vocab.items())
    assert merges == [tuple(v.split(" ")) for v in m["tokenizer.ggml.merges"]]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    for token in hf["added_tokens"]:
        tk.add_tokens([AddedToken(**{k: v for k, v in token.items() if k != "id"})])
        assert tk.token_to_id(token["content"]) == token["id"]
        assert m["tokenizer.ggml.tokens"][token["id"]] == token["content"]

    texts = [
        "Hello!", "你好，请用一句话介绍自己。", " Hello  world\t\r\n\n  ",
        "I'm I'M we're can't won't I'd", "1234567890 3.1415926 -42 1e10",
        "café cafe\u0301 a\u035cb हिन्दी العربية", "🙂👩🏽‍💻🚀🇨🇳", "a\x00b",
        "def f(x):\n    return x + 1\n", "<|im_start|>user\nHello!<|im_end|>\n",
        "<think>分析</think>\n\n答案<|im_end|>", "<tool_call>{}</tool_call>",
    ]
    encodings = []
    for text in texts:
        ids = tk.encode(text, add_special_tokens=False).ids
        encodings.append({"text": text, "ids": ids, "decoded": tk.decode(ids, skip_special_tokens=False)})

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    def raise_exception(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(m["tokenizer.chat_template"])
    user = {"role": "user", "content": "你好，请用一句话介绍自己。"}
    cases = [
        ("no_thinking", [user], False, None),
        ("default_thinking", [user], True, None),
        ("low_effort", [user], True, "low"),
        ("medium_effort", [user], True, "medium"),
        ("high_alias", [user], True, "high"),
        ("merged_system", [{"role": "system", "content": "  简洁回答。 "}, {"role": "developer", "content": "用中文。"}, user], False, None),
        ("history", [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2", "reasoning_content": "addition"}, user], True, "medium"),
    ]
    chats = []
    for name, messages, thinking, effort in cases:
        context = dict(messages=messages, enable_thinking=thinking, add_generation_prompt=True)
        if effort is not None:
            context["reasoning_effort"] = effort
        rendered = template.render(**context)
        chats.append(dict(name=name, messages=messages, thinking=thinking, effort=effort,
                          rendered=rendered, ids=tk.encode(rendered, add_special_tokens=False).ids))
    result = dict(
        profile_source=PROFILE_SOURCE, pattern=PATTERN,
        tokenizers_version=tokenizers.__version__, jinja2_version=jinja2.__version__,
        hf_tokenizer_sha256=hashlib.sha256(args.hf_tokenizer.read_bytes()).hexdigest(),
        chat_template_sha256=hashlib.sha256(m["tokenizer.chat_template"].encode()).hexdigest(),
        encodings=encodings, chats=chats,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(encodings)} encoding and {len(chats)} template cases to {args.output}")


if __name__ == "__main__":
    main()
