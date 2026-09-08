"""Run Qwen3.5 image HTTP regression against a temporary three-process stack.

Export references with qwen35_vision_precision.py first. The supplied TOML must
point at Qwen3.5 and an available CUDA device; image prefill uses eager execution.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import pathlib
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=pathlib.Path, required=True)
parser.add_argument('--reference', type=pathlib.Path, required=True)
parser.add_argument('--bin-dir', type=pathlib.Path)
parser.add_argument('--require-decode-graphs', action='store_true', help='Assert image decode replay in worker logs; set log_level="debug" in the config')
args = parser.parse_args()
config = tomllib.loads(args.config.read_text())
base_url = f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 8100)}"
root = pathlib.Path(__file__).resolve().parent.parent
bin_dir = args.bin_dir or root / 'target/debug'
logs = pathlib.Path(tempfile.mkdtemp(prefix='qwen35-mm-http-'))
print('Logs:', logs, flush=True)
processes = []
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = 'data:image/png;base64,' + base64.b64encode((args.reference / 'image.png').read_bytes()).decode()
image = {'type': 'image_url', 'image_url': {'url': url}}
text = {'type': 'text', 'text': 'Describe the image briefly.'}
jpeg = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode((args.reference / 'image.jpg').read_bytes()).decode()}}

def body(parts):
    return dict(model='Qwen3.5-4B', messages=[dict(role='user', content=parts)], max_tokens=8, temperature=0, ignore_eos=True)

def request(payload):
    req = urllib.request.Request(base_url + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=120) as r:
        if payload.get('stream'):
            raw = r.read().decode()
            assert 'data: [DONE]' in raw and 'event: error' not in raw, raw
            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            return ''.join((c['choices'][0]['delta'].get('content', '') for c in chunks if c.get('choices')))
        result = json.load(r)
        return result['choices'][0]['message']['content']
try:
    for name in ['scheduler', 'worker', 'server']:
        processes.append(subprocess.Popen([str(bin_dir / ('rustinfer-' + name)), '--config', str(args.config.resolve())], stdout=open(logs / (name + '.log'), 'w'), stderr=subprocess.STDOUT))
    for _ in range(180):
        if any((p.poll() is not None for p in processes)):
            raise RuntimeError('process exited: ' + str(logs))
        if 'Entering serve loop' in (logs / 'worker.log').read_text():
            try:
                opener.open(base_url + '/health', timeout=1).close()
                break
            except urllib.error.URLError:
                pass
        time.sleep(1)
    else:
        raise RuntimeError('readiness timeout')
    base = request(body([image, text]))
    expected = json.loads((args.reference / 'metadata.json').read_text())['text']
    print('image response:', repr(base), flush=True)
    assert base == expected, (base, expected)
    assert request(body([image, text])) == base
    assert request(dict(body([image, text]), stream=True)) == base
    cases = [body([image, text]), body('The capital of France is'), body([image, text, image]), body([{'type': 'text', 'text': 'Look carefully. '}, image, text])]
    cases.append(body([jpeg, text]))
    baseline = [request(p) for p in cases]
    for _ in range(2):
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            actual = list(pool.map(request, cases))
        assert actual == baseline, (actual, baseline)
    # Unequal generation lengths force finished-row compaction while image
    # and text requests continue decoding together, including streaming output.
    varied = [dict(p, max_tokens=n, stream=(i % 2 == 0)) for i, (p, n) in enumerate(zip(cases[:4], [3, 17, 11, 7]))]
    varied_baseline = [request(p) for p in varied]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert list(pool.map(request, varied)) == varied_baseline
    for invalid in [[{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}}], [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}], 'what is <|image_pad|>', [image] * 5]:
        try:
            request(body(invalid))
            raise AssertionError('invalid image accepted')
        except urllib.error.HTTPError as e:
            assert e.code == 400, (e.code, e.read())
    req = urllib.request.Request(base_url + '/v1/chat/completions', data=json.dumps(dict(body([image, text]), stream=True, max_tokens=128)).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=120) as r:
        r.readline()
    time.sleep(0.2)
    assert request(body([image, text])) == base
    if args.require_decode_graphs:
        import re
        worker_log = re.sub(r'\x1b\[[0-9;]*m', '', (logs / 'worker.log').read_text())
        replays = [line for line in worker_log.splitlines() if 'replaying decode CUDA graph' in line and 'multimodal=true' in line]
        assert replays, 'No image decode graph replay found; check capture_sizes and debug logging'
        print(f'image decode CUDA Graph replays: {len(replays)}', flush=True)
    print('PASS: image HF match; repeat/cache; SSE; two images; mixed concurrency; invalid inputs; cancellation', flush=True)
finally:
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
