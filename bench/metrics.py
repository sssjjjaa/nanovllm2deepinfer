"""报告的汇总逻辑。纯 Python，不 import torch。"""
SEGMENTS = [
    "cpu_schedule",
    "cpu_prepare",
    "cpu_launch",
    "cpu_sample_wait",
    "gpu_forward",
    "gpu_sample",
    "cpu_postprocess",
]


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数。单元素直接返回，空列表返回 0。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _agg(values: list[float]) -> dict:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0.0,
    }


def summarize_steps(records: list) -> dict:
    """按 prefill / decode 分组统计每一步的 7 段耗时。

    gpu_busy 必须按时间加权：总 GPU 时间除以总墙钟，
    不是对每步的比例求平均——30 ms 的步和 10 ms 的步分量不一样。
    """
    out = {
        "num_steps": len(records),
        "preemptions": sum(len(r.preempted) for r in records),
    }
    for group, is_prefill in (("prefill", True), ("decode", False)):
        rs = [r for r in records if r.is_prefill == is_prefill]
        d = {
            "steps": len(rs),
            "tokens": sum(r.num_tokens for r in rs),
            "segments": {},
        }
        if rs:
            wall = sum(r.wall for r in rs)
            gpu = sum(r.gpu_forward + r.gpu_sample for r in rs)
            d["gpu_busy"] = gpu / wall if wall > 0 else 0.0
            d["wall"] = _agg([r.wall for r in rs])
            d["batch"] = _agg([float(r.num_seqs) for r in rs])
            for key in SEGMENTS:
                d["segments"][key] = _agg([getattr(r, key) for r in rs])
        out[group] = d
    return out


def summarize_requests(records: list) -> dict:
    return {
        "num_requests": len(records),
        "completion_tokens": sum(r.num_completion_tokens for r in records),
        "ttft": _agg([r.ttft for r in records]),
        "tbt": _agg([x for r in records for x in r.tbt]),
        "latency": _agg([r.latency for r in records]),
        "preempted_requests": sum(1 for r in records if r.num_preemptions),
        "preemptions": sum(r.num_preemptions for r in records),
    }


def _ms(x: float) -> str:
    return f"{x * 1e3:.2f}"


def format_report(summary: dict) -> str:
    lines = []
    steps = summary["steps"]
    requests = summary["requests"]
    lines.append(f"steps={steps['num_steps']}  preemption={steps['preemptions']}")
    for group in ("prefill", "decode"):
        d = steps[group]
        if not d["steps"]:
            continue
        lines.append(f"[{group}] steps={d['steps']} tokens={d['tokens']} "
                     f"gpu_busy={d.get('gpu_busy', 0) * 100:.1f}% "
                     f"bs_p50={d['batch']['p50']:.0f}")
        lines.append(f"  {'':<18}{'p50':>9}{'p99':>9}{'max':>9}  (ms)")
        lines.append(f"  {'wall':<18}{_ms(d['wall']['p50']):>9}{_ms(d['wall']['p99']):>9}{_ms(d['wall']['max']):>9}")
        for key in SEGMENTS:
            s = d["segments"][key]
            lines.append(f"  {key:<18}{_ms(s['p50']):>9}{_ms(s['p99']):>9}{_ms(s['max']):>9}")
    lines.append(f"[requests] n={requests['num_requests']} tokens={requests['completion_tokens']}")
    lines.append(f"  ttft(ms)  p50={_ms(requests['ttft']['p50'])} p99={_ms(requests['ttft']['p99'])} "
                 f"max={_ms(requests['ttft']['max'])}")
    lines.append(f"  tbt(ms)   p50={_ms(requests['tbt']['p50'])} p99={_ms(requests['tbt']['p99'])} "
                 f"max={_ms(requests['tbt']['max'])}")
    lines.append(f"  latency(s) p50={requests['latency']['p50']:.2f} p99={requests['latency']['p99']:.2f}")
    lines.append(f"  preempted={requests['preempted_requests']} 个请求 / {requests['preemptions']} 次")
    return "\n".join(lines)
