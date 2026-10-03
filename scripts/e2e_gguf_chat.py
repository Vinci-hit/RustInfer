"""GGUF chat regression against a running RustInfer worker/scheduler/server.

Uses the standard library. Start the stack with rustinfer.toml, then run:
python3 scripts/e2e_gguf_chat.py --url http://127.0.0.1:8080
The test submits real GPU requests, including a cancelled stream.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=Path("/tmp/gguf-service-results.json"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    results = {}

    def request(path, data=None):
        return urllib.request.urlopen(urllib.request.Request(
            base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}), timeout=180)

    def post(data, path="/v1/chat/completions"):
        with request(path, data) as response:
            return json.load(response)

    def chat(messages=None, **kwargs):
        return {"messages": messages or [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
                "enable_thinking": False, "temperature": 0, "max_tokens": 64, **kwargs}

    def stream(data, cancel=False):
        text, finish, usage, done, events = "", None, None, False, 0
        with request("/v1/chat/completions", data) as response:
            for line in response:
                assert not line.startswith(b"event: error"), line
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(payload)
                assert "error" not in chunk, chunk
                events += 1
                for choice in chunk.get("choices", []):
                    delta = choice["delta"].get("content") or ""
                    text += delta
                    finish = choice.get("finish_reason") or finish
                    if cancel and delta:
                        return {"cancelled_after": text}
                usage = chunk.get("usage") or usage
        assert done and finish and usage, (done, finish, usage)
        assert "\ufffd" not in text
        return {"text": text, "finish_reason": finish, "usage": usage, "events": events}

    started = time.monotonic()
    try:
        with request("/ready") as response:
            assert response.status == 200
        with request("/v1/models") as response:
            results["models"] = json.load(response)
        with request("/v1/capabilities") as response:
            caps = json.load(response)
        assert caps["thinking"] and caps["inputs"]["text"] and not caps["inputs"]["image"]
        results["capabilities"] = caps
        baseline = post(chat())
        text = baseline["choices"][0]["message"]["content"]
        assert text and baseline["choices"][0]["finish_reason"] == "stop"
        results["nonstream"] = baseline
        streamed = stream(chat(stream=True, stream_options={"include_usage": True}))
        assert streamed["text"] == text
        assert streamed["usage"]["completion_tokens"] == baseline["usage"]["completion_tokens"]
        results["stream"] = streamed
        # Prompt exceeds the configured 64-token prefill chunk, and includes
        # assistant history so it exercises the GGUF conversation template.
        messages = [{"role": "system", "content": "你是一个简洁的助手。" * 30},
                    {"role": "user", "content": "记住暗号是蓝鲸。"},
                    {"role": "assistant", "content": "记住了，暗号是蓝鲸。"},
                    {"role": "user", "content": "刚才的暗号是什么？只回答暗号。"}]
        history = post(chat(messages, max_tokens=32))
        assert history["usage"]["prompt_tokens"] > 64
        assert "蓝鲸" in history["choices"][0]["message"]["content"], history
        results["chunked_history"] = history
        results["thinking"] = post(chat(enable_thinking=True, reasoning_effort="low", max_tokens=8))
        assert results["thinking"]["choices"][0]["finish_reason"] == "length"
        results["sampling"] = post(chat(temperature=0.7, top_p=0.95, max_tokens=8))
        results["stop"] = post(chat(stop="通义"))
        assert "通义" not in results["stop"]["choices"][0]["message"]["content"]
        assert results["stop"]["choices"][0]["finish_reason"] == "stop"
        results["raw_completion"] = post({"prompt": "你好", "temperature": 0, "max_tokens": 4}, "/v1/completions")
        results["cancel"] = stream(chat(stream=True, ignore_eos=True, max_tokens=512), cancel=True)
        # A cancelled request must release its sole recurrent slot/KV lease.
        after = post(chat())
        assert after["choices"][0]["message"]["content"] == text
        results["after_cancel"] = after
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            queued = list(pool.map(lambda _: post(chat(max_tokens=4)), range(2)))
        assert all(r["choices"][0]["message"]["content"] == queued[0]["choices"][0]["message"]["content"] for r in queued)
        results["queued_requests"] = queued
        invalid = [chat(reasoning_effort="invalid"),
                   chat([{"role":"tool", "content":"unsupported"}]),
                   chat([{"role":"user", "content":"你好 " * 3000}]),
                   chat([{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AAAA"}}]}])]
        for data in invalid:
            try:
                post(data)
                raise AssertionError("invalid request accepted")
            except urllib.error.HTTPError as error:
                assert error.code == 400, error.read()
        results["rejected_requests"] = len(invalid)
        results["ok"] = True
    finally:
        results["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"results: {args.output}", flush=True)
    print("GGUF service regression passed", flush=True)


if __name__ == "__main__":
    main()
