"""
真实到达时间 online benchmark。

特点：
- 使用 bench_prompts.json 真实 prompt 数据
- 支持短/中/长/mix prompt 长度场景
- 支持多 arrival rate 扫描（Poisson inter-arrival，固定 seed 可复现）
- warmup 请求不计入统计
- 对 OpenAI-compatible `/v1/chat/completions` 服务通用，可用于 RustInfer / vLLM 对比
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp


@dataclass
class RequestSpec:
    request_id: int
    arrival_s: float
    prompt: str
    max_tokens: int


@dataclass
class RequestResult:
    request_id: int
    arrival_s: float
    prompt_words: int
    status: str
    output_tokens: int
    latency_ms: float
    ttft_ms: Optional[float]
    error: Optional[str] = None


@dataclass
class ServerConfig:
    url: str
    label: str
    model: str


def percentile(values, p):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(int(len(xs) * p), len(xs) - 1)
    return xs[idx]


def mean_or_zero(values) -> float:
    xs = list(values)
    return statistics.mean(xs) if xs else 0.0


def default_prompts_path() -> str:
    return str(Path(__file__).with_name("bench_prompts.json"))


def load_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"prompts file must contain a JSON list: {path}")
    prompts = [p.strip() for p in data if isinstance(p, str) and p.strip()]
    if not prompts:
        raise ValueError(f"prompts file has no usable prompts: {path}")
    return prompts


def bucket_prompts(prompts: list[str]) -> dict[str, list[str]]:
    buckets = {
        "short": [],
        "medium": [],
        "long": [],
        "mix": list(prompts),
    }
    for prompt in prompts:
        words = len(prompt.split())
        if words <= 15:
            buckets["short"].append(prompt)
        elif words <= 50:
            buckets["medium"].append(prompt)
        else:
            buckets["long"].append(prompt)
    return buckets


def parse_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_arrival_rates(args) -> list[float]:
    raw_rates = parse_csv(args.arrival_rates)
    if not raw_rates:
        return [args.arrival_rate]
    rates = []
    for raw in raw_rates:
        try:
            rate = float(raw)
        except ValueError as e:
            raise ValueError(f"invalid arrival rate: {raw}") from e
        if rate < 0:
            raise ValueError(f"arrival rate must be >= 0: {raw}")
        rates.append(rate)
    if not rates:
        raise ValueError("at least one arrival rate is required")
    return rates


def parse_length_buckets(value: str, buckets: dict[str, list[str]]) -> list[str]:
    selected = parse_csv(value)
    if not selected:
        raise ValueError("at least one length bucket is required")
    valid = set(buckets)
    for bucket in selected:
        if bucket not in valid:
            raise ValueError(f"unknown length bucket: {bucket}; valid={sorted(valid)}")
        if not buckets[bucket]:
            raise ValueError(f"selected length bucket has no prompts: {bucket}")
    return selected


def expand_to_count(values: list[str], count: int, name: str) -> list[str]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) == count:
        return values
    if len(values) == 1:
        return values * count
    raise ValueError(f"{name} count ({len(values)}) must be 1 or match urls count ({count})")


def resolve_servers(args) -> list[ServerConfig]:
    urls = parse_csv(args.urls) or [args.url]
    labels = parse_csv(args.labels) or [args.label]
    models = parse_csv(args.models) or [args.model]

    labels = expand_to_count(labels, len(urls), "labels")
    models = expand_to_count(models, len(urls), "models")
    return [ServerConfig(url=url.rstrip("/"), label=label, model=model) for url, label, model in zip(urls, labels, models)]


def make_specs(
    num_requests: int,
    arrival_rate: float,
    max_tokens: int,
    seed: int,
    prompts_pool: list[str],
) -> list[RequestSpec]:
    rng = random.Random(seed)
    specs = []
    t = 0.0
    for i in range(num_requests):
        if i > 0 and arrival_rate > 0:
            # Poisson arrival: exponential inter-arrival，固定 seed 可复现。
            t += rng.expovariate(arrival_rate)
        specs.append(RequestSpec(i, t, rng.choice(prompts_pool), max_tokens))
    return specs


async def health_check(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"health check failed for {url}: HTTP {resp.status}")


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    spec: RequestSpec,
    stream: bool,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "min_tokens": spec.max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": stream,
    }
    start = time.perf_counter()
    prompt_words = len(spec.prompt.split())
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if stream:
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                                         (time.perf_counter() - start) * 1000.0, None, body[:200])
                first_token_time = None
                output_tokens = 0
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        output_tokens += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                end = time.perf_counter()
                ttft_ms = (first_token_time - start) * 1000.0 if first_token_time else None
                return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "ok", output_tokens,
                                     (end - start) * 1000.0, ttft_ms)

            body = await resp.json()
            end = time.perf_counter()
            if resp.status != 200:
                return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                                     (end - start) * 1000.0, None, str(body)[:200])
            usage = body.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or 0)
            return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "ok", output_tokens,
                                 (end - start) * 1000.0, None)
    except Exception as e:
        return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                             (time.perf_counter() - start) * 1000.0, None, repr(e)[:200])


async def run_specs(url: str, model: str, specs: list[RequestSpec], concurrency: int, stream: bool, verbose: bool):
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    start = time.perf_counter()

    async def one(spec: RequestSpec):
        delay = start + spec.arrival_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                r = await send_request(session, url, model, spec, stream)
            results.append(r)
            if verbose:
                mark = "ok" if r.status == "ok" else "ERR"
                print(f"  [{mark}] req-{r.request_id:03d} arrive={r.arrival_s:7.3f}s "
                      f"prompt_words={r.prompt_words:3d} out={r.output_tokens:3d} "
                      f"lat={r.latency_ms:8.1f}ms")

    await asyncio.gather(*(asyncio.create_task(one(s)) for s in specs))
    wall_s = time.perf_counter() - start
    return sorted(results, key=lambda r: r.request_id), wall_s


def summarize(label: str, results: list[RequestResult], wall_s: float):
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    lat = [r.latency_ms for r in ok]
    prompt_words = [r.prompt_words for r in ok]
    toks = sum(r.output_tokens for r in ok)
    tpots = [(r.latency_ms / r.output_tokens) for r in ok if r.output_tokens > 0]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]

    print("\n" + "=" * 72)
    print(f"  {label} results")
    print("=" * 72)
    print(f"  wall time:       {wall_s:.3f} s")
    print(f"  requests:        {len(results)} total, {len(ok)} ok, {len(failed)} failed")
    print(f"  output tokens:   {toks}")
    print(f"  throughput:      {(toks / wall_s) if wall_s > 0 else 0:.1f} tok/s")
    if ok:
        print(f"  prompt words:    mean={statistics.mean(prompt_words):.1f}, "
              f"min={min(prompt_words)}, max={max(prompt_words)}")
        print(f"  latency ms:      mean={statistics.mean(lat):.1f}, p50={percentile(lat, 0.50):.1f}, "
              f"p90={percentile(lat, 0.90):.1f}, p99={percentile(lat, 0.99):.1f}")
        req_tok_s = [r.output_tokens / (r.latency_ms / 1000.0) for r in ok if r.latency_ms > 0]
        print(f"  req tok/s:       mean={mean_or_zero(req_tok_s):.1f}")
        if tpots:
            print(f"  latency/out tok: mean={statistics.mean(tpots):.3f} ms, p50={percentile(tpots, 0.50):.3f}, "
                  f"p90={percentile(tpots, 0.90):.3f}")
        if ttfts:
            print(f"  TTFT ms:         mean={statistics.mean(ttfts):.1f}, p50={percentile(ttfts, 0.50):.1f}, "
                  f"p90={percentile(ttfts, 0.90):.1f}, p99={percentile(ttfts, 0.99):.1f}")
    if failed:
        print("  failures:")
        for r in failed[:5]:
            print(f"    req-{r.request_id}: {r.error}")
    print("=" * 72)
    return {
        "requests": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "error_rate": (len(failed) / len(results)) if results else 0,
        "tokens": toks,
        "throughput": toks / wall_s if wall_s > 0 else 0,
        "prompt_words_mean": statistics.mean(prompt_words) if prompt_words else 0,
        "prompt_words_min": min(prompt_words) if prompt_words else 0,
        "prompt_words_max": max(prompt_words) if prompt_words else 0,
        "lat_mean": statistics.mean(lat) if lat else 0,
        "lat_p50": percentile(lat, 0.50),
        "lat_p90": percentile(lat, 0.90),
        "lat_p99": percentile(lat, 0.99),
        "tpot_mean": statistics.mean(tpots) if tpots else 0,
        "tpot_p50": percentile(tpots, 0.50),
        "tpot_p90": percentile(tpots, 0.90),
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0,
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p90": percentile(ttfts, 0.90),
        "ttft_p99": percentile(ttfts, 0.99),
    }


def print_prompt_stats(buckets: dict[str, list[str]]):
    print("Prompt buckets:")
    for name in ("short", "medium", "long", "mix"):
        words = [len(p.split()) for p in buckets[name]]
        if not words:
            print(f"  {name:6s}: 0 prompts")
            continue
        print(f"  {name:6s}: {len(words):6d} prompts, words mean={statistics.mean(words):5.1f}, "
              f"min={min(words):3d}, max={max(words):3d}")


def print_comparison_table(entries: list[dict[str, Any]]):
    if not entries:
        return

    print("\n" + "#" * 96)
    print("Final comparison")
    print("#" * 96)
    groups = sorted({(e["arrival_rate"], e["length_bucket"]) for e in entries}, key=lambda x: (x[0], x[1]))
    for arrival_rate, length_bucket in groups:
        print(f"\nRPS={arrival_rate:g} | len={length_bucket}")
        print(f"{'label':14s} {'tok/s':>10s} {'lat_avg':>10s} {'lat_p50':>10s} {'lat_p90':>10s} "
              f"{'lat_p99':>10s} {'ttft_p90':>10s} {'tpot_avg':>10s} {'err%':>8s}")
        for entry in [e for e in entries if e["arrival_rate"] == arrival_rate and e["length_bucket"] == length_bucket]:
            s = entry["summary"]
            print(f"{entry['label']:14s} {s['throughput']:10.1f} {s['lat_mean']:10.1f} {s['lat_p50']:10.1f} "
                  f"{s['lat_p90']:10.1f} {s['lat_p99']:10.1f} {s['ttft_p90']:10.1f} "
                  f"{s['tpot_mean']:10.3f} {s['error_rate'] * 100:8.2f}")


def write_json_report(path: str, args, prompt_counts: dict[str, int], entries: list[dict[str, Any]]):
    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "prompt_counts": prompt_counts,
        },
        "results": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\nWrote JSON report: {path}")


def plot_metric_by_rps_png(entries: list[dict[str, Any]], plot_dir: Path, metric: str, title: str, ylabel: str):
    import matplotlib.pyplot as plt

    buckets = sorted({e["length_bucket"] for e in entries})
    labels = sorted({e["label"] for e in entries})
    fig, axes = plt.subplots(1, len(buckets), figsize=(6 * len(buckets), 4), squeeze=False)

    for ax, bucket in zip(axes[0], buckets):
        for label in labels:
            rows = sorted(
                [e for e in entries if e["length_bucket"] == bucket and e["label"] == label],
                key=lambda e: e["arrival_rate"],
            )
            if not rows:
                continue
            xs = [e["arrival_rate"] for e in rows]
            ys = [e["summary"].get(metric, 0.0) for e in rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        ax.set_title(f"{bucket} prompts")
        ax.set_xlabel("Arrival rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = plot_dir / f"{metric}_vs_rps.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial">{escaped}</text>'


def plot_metric_by_rps_svg(entries: list[dict[str, Any]], plot_dir: Path, metric: str, title: str, ylabel: str):
    buckets = sorted({e["length_bucket"] for e in entries})
    labels = sorted({e["label"] for e in entries})
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    panel_w, panel_h = 460, 340
    margin_l, margin_r, margin_t, margin_b = 62, 22, 52, 58
    width, height = panel_w * len(buckets), panel_h + 72
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(svg_text(width / 2, 28, title, 18))

    for i, bucket in enumerate(buckets):
        x0 = i * panel_w
        rows_by_label = {
            label: sorted(
                [e for e in entries if e["length_bucket"] == bucket and e["label"] == label],
                key=lambda e: e["arrival_rate"],
            )
            for label in labels
        }
        xs_all = [e["arrival_rate"] for rows in rows_by_label.values() for e in rows]
        ys_all = [e["summary"].get(metric, 0.0) for rows in rows_by_label.values() for e in rows]
        if not xs_all:
            continue
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = 0.0, max(ys_all) if ys_all else 0.0
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        plot_x0, plot_y0 = x0 + margin_l, margin_t + 20
        plot_w, plot_h = panel_w - margin_l - margin_r, panel_h - margin_t - margin_b

        def sx(x):
            return plot_x0 + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y):
            return plot_y0 + plot_h - (y - y_min) / (y_max - y_min) * plot_h

        svg.append(svg_text(x0 + panel_w / 2, 55, f"{bucket} prompts", 14))
        svg.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#333"/>')
        svg.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#333"/>')
        for tick in range(5):
            y_val = y_min + (y_max - y_min) * tick / 4
            y = sy(y_val)
            svg.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x0 + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-dasharray="3,3"/>')
            svg.append(svg_text(plot_x0 - 8, y + 4, f"{y_val:.1f}", 10, "end"))
        for x_val in sorted(set(xs_all)):
            x = sx(x_val)
            svg.append(f'<line x1="{x:.1f}" y1="{plot_y0 + plot_h}" x2="{x:.1f}" y2="{plot_y0 + plot_h + 5}" stroke="#333"/>')
            svg.append(svg_text(x, plot_y0 + plot_h + 20, f"{x_val:g}", 10))
        svg.append(svg_text(plot_x0 + plot_w / 2, plot_y0 + plot_h + 42, "Arrival rate (req/s)", 11))
        svg.append(svg_text(plot_x0 - 44, plot_y0 + plot_h / 2, ylabel, 11, "middle"))

        for label_idx, label in enumerate(labels):
            rows = rows_by_label[label]
            if not rows:
                continue
            color = colors[label_idx % len(colors)]
            points = [(sx(e["arrival_rate"]), sy(e["summary"].get(metric, 0.0))) for e in rows]
            point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            svg.append(f'<polyline points="{point_attr}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            legend_x = plot_x0 + 10
            legend_y = plot_y0 + 16 + label_idx * 18
            svg.append(f'<rect x="{legend_x}" y="{legend_y - 9}" width="12" height="3" fill="{color}"/>')
            svg.append(svg_text(legend_x + 18, legend_y - 5, label, 10, "start"))

    svg.append("</svg>")
    path = plot_dir / f"{metric}_vs_rps.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_comparison(entries: list[dict[str, Any]], plot_dir: str):
    if not entries:
        return

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("throughput", "Throughput vs arrival rate", "Output throughput (tok/s)"),
        ("lat_p50", "P50 latency vs arrival rate", "Latency p50 (ms)"),
        ("lat_p90", "P90 latency vs arrival rate", "Latency p90 (ms)"),
        ("lat_p99", "P99 latency vs arrival rate", "Latency p99 (ms)"),
        ("ttft_p90", "P90 TTFT vs arrival rate", "TTFT p90 (ms)"),
        ("tpot_mean", "Mean TPOT vs arrival rate", "TPOT mean (ms/output token)"),
        ("error_rate", "Error rate vs arrival rate", "Error rate"),
    ]
    try:
        import matplotlib  # noqa: F401
        plotter = plot_metric_by_rps_png
        plot_type = "PNG"
    except ImportError:
        plotter = plot_metric_by_rps_svg
        plot_type = "SVG"

    paths = [plotter(entries, out_dir, metric, title, ylabel) for metric, title, ylabel in metrics]
    print(f"\nWrote benchmark {plot_type} plots:")
    for path in paths:
        print(f"  {path}")


async def main_async(args):
    prompts = load_prompts(args.prompts_file)
    buckets = bucket_prompts(prompts)
    length_buckets = parse_length_buckets(args.length_buckets, buckets)
    arrival_rates = parse_arrival_rates(args)
    servers = resolve_servers(args)

    print(f"Prompts file:    {args.prompts_file}")
    print_prompt_stats(buckets)
    print(f"Selected buckets: {', '.join(length_buckets)}")
    print(f"Arrival rates:   {', '.join(f'{r:g}' for r in arrival_rates)} req/s")
    print(f"Servers:         {', '.join(f'{s.label}={s.url}' for s in servers)}")
    print(f"Bench requests:  {args.num_requests}")
    print(f"Warmup requests: {args.warmup_requests}")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Max tokens:      {args.max_tokens}")
    print(f"Stream:          {args.stream}")
    print(f"Seed:            {args.seed}")

    for server in servers:
        await health_check(server.url)

    entries: list[dict[str, Any]] = []
    for rate_idx, arrival_rate in enumerate(arrival_rates):
        for bucket_idx, bucket_name in enumerate(length_buckets):
            pool = buckets[bucket_name]
            scenario_seed = args.seed + rate_idx * 100003 + bucket_idx * 1009
            warmup_specs = make_specs(args.warmup_requests, arrival_rate, args.max_tokens, scenario_seed + 1000, pool)
            bench_specs = make_specs(args.num_requests, arrival_rate, args.max_tokens, scenario_seed, pool)

            print("\n" + "#" * 96)
            print(f"Scenario: arrival_rate={arrival_rate:g} req/s, length_bucket={bucket_name}, prompts={len(pool)}")
            print("#" * 96)

            for server in servers:
                scenario_label = f"{server.label} | rps={arrival_rate:g} | len={bucket_name}"
                print(f"\nService:        {server.label}")
                print(f"URL:            {server.url}")
                print(f"Model:          {server.model}")

                if args.warmup_requests > 0:
                    print("\nWarmup phase...")
                    warmup_results, warmup_wall = await run_specs(
                        server.url, server.model, warmup_specs, args.concurrency, args.stream, args.verbose
                    )
                    summarize(f"{scenario_label} warmup (not counted)", warmup_results, warmup_wall)

                print("\nBenchmark phase...")
                results, wall_s = await run_specs(
                    server.url, server.model, bench_specs, args.concurrency, args.stream, args.verbose
                )
                summary = summarize(scenario_label, results, wall_s)
                entry = {
                    "label": server.label,
                    "url": server.url,
                    "model": server.model,
                    "arrival_rate": arrival_rate,
                    "length_bucket": bucket_name,
                    "summary": summary,
                }
                entries.append(entry)
                print("JSON_SUMMARY " + json.dumps(entry, sort_keys=True))

    print_comparison_table(entries)

    if not args.no_plots:
        plot_comparison(entries, args.plot_dir)

    if args.output_json:
        prompt_counts = {name: len(values) for name, values in buckets.items()}
        write_json_report(args.output_json, args, prompt_counts, entries)


def main():
    parser = argparse.ArgumentParser(description="Real-arrival OpenAI-compatible online benchmark")
    parser.add_argument("--prompts-file", default=default_prompts_path())

    parser.add_argument("--url", default="http://127.0.0.1:8000", help="single-server URL alias")
    parser.add_argument("--model", default="checkpoint-800-1", help="single-server model alias")
    parser.add_argument("--label", default="server", help="single-server label alias")
    parser.add_argument("--arrival-rate", type=float, default=24.0, help="single arrival rate alias")

    parser.add_argument("--urls", default=None, help="comma-separated server URLs, e.g. rustinfer,vllm endpoints")
    parser.add_argument("--labels", default=None, help="comma-separated labels matching --urls")
    parser.add_argument("--models", default=None, help="comma-separated model names matching --urls")
    parser.add_argument("--arrival-rates", default=None, help="comma-separated req/s values, e.g. 4,8,16,24")
    parser.add_argument("--length-buckets", default="short,medium,long,mix", help="comma-separated: short,medium,long,mix")
    parser.add_argument("--output-json", default=None, help="optional path for full JSON report")
    parser.add_argument("--plot-dir", default="bench_plots", help="directory for benchmark comparison plots")
    parser.add_argument("--no-plots", action="store_true", help="disable benchmark plot generation")

    parser.add_argument("--num-requests", type=int, default=80)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
